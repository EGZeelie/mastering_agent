"""
The DSP Control Loop ("The Hands")
====================================

A headless, serial mastering chain operating on float64 stereo buffers:

    linear-phase EQ -> surgical dynamic EQ -> VCA-style bus compressor
    -> harmonic saturation -> true-peak brickwall limiter

Every stage is driven purely by a `ChainParams` dataclass so the decision
core (LLM or rule-based) never touches DSP code directly -- it only ever
emits parameter deltas, which keeps the "brain" and "hands" cleanly
decoupled (a prerequisite for the iterative render/re-analyze loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import fftconvolve, resample_poly


# ---------------------------------------------------------------------------
# Parameter contracts
# ---------------------------------------------------------------------------

@dataclass
class EQBand:
    freq_hz: float
    gain_db: float
    q: float = 1.0
    kind: str = "bell"  # "bell", "low_shelf", "high_shelf"


@dataclass
class DynamicEQBand:
    freq_hz: float
    q: float
    threshold_db: float
    ratio: float
    max_cut_db: float   # negative-only surgical cut, e.g. de-essing / resonance taming
    attack_ms: float = 5.0
    release_ms: float = 80.0


@dataclass
class CompressorParams:
    threshold_db: float = -18.0
    ratio: float = 2.0
    attack_ms: float = 15.0
    release_ms: float = 120.0
    makeup_db: float = 0.0
    mix: float = 1.0  # 1.0 = fully in-line, <1.0 = parallel blend


@dataclass
class SaturationParams:
    drive_db: float = 0.0    # pre-gain into the nonlinearity
    mix: float = 0.0         # 0 = bypass, 1 = fully saturated
    kind: str = "tanh"


@dataclass
class LimiterParams:
    ceiling_dbtp: float = -1.0
    release_ms: float = 60.0
    lookahead_ms: float = 5.0


@dataclass
class ChainParams:
    linear_eq: List[EQBand] = field(default_factory=list)
    dynamic_eq: List[DynamicEQBand] = field(default_factory=list)
    compressor: CompressorParams = field(default_factory=CompressorParams)
    saturation: SaturationParams = field(default_factory=SaturationParams)
    limiter: LimiterParams = field(default_factory=LimiterParams)
    input_trim_db: float = 0.0
    stereo_width_mult: float = 1.0  # 1.0 = unchanged, >1 wider, <1 narrower


# ---------------------------------------------------------------------------
# Linear-phase EQ (FIR, frequency-sampling design)
# ---------------------------------------------------------------------------

def _peaking_response(freqs, f0, gain_db, q):
    """Analog-prototype peaking filter magnitude response, evaluated at freqs (Hz)."""
    A = 10 ** (gain_db / 40.0)
    w0 = f0
    bw = w0 / max(q, 0.05)
    # simple resonant bump/dip approximated in log-frequency space (Gaussian-ish bell)
    log_f = np.log2(np.maximum(freqs, 1.0))
    log_f0 = np.log2(max(w0, 1.0))
    log_bw_oct = np.log2(1 + bw / max(w0, 1.0))
    sigma = max(log_bw_oct, 1e-3)
    bump = (A - 1) * np.exp(-0.5 * ((log_f - log_f0) / sigma) ** 2)
    return 1.0 + bump


def _shelf_response(freqs, f0, gain_db, kind):
    A = 10 ** (gain_db / 20.0)
    log_f = np.log2(np.maximum(freqs, 1.0))
    log_f0 = np.log2(max(f0, 1.0))
    x = (log_f - log_f0)
    sigmoid = 1 / (1 + np.exp(-2.5 * x))
    if kind == "high_shelf":
        return 1.0 + (A - 1.0) * sigmoid
    else:  # low_shelf
        return 1.0 + (A - 1.0) * (1.0 - sigmoid)


def design_linear_phase_eq(bands: List[EQBand], sr: int, numtaps: int = 2049) -> np.ndarray:
    """Build a linear-phase FIR filter approximating the sum of all EQ bands
    via frequency sampling + windowing (Kaiser). Linear phase => zero phase
    distortion, at the cost of latency (numtaps/2 samples), which is the
    standard mastering-grade tradeoff."""
    n_freqs = numtaps
    freqs = np.linspace(0, sr / 2, n_freqs)
    response = np.ones_like(freqs)
    for b in bands:
        if b.kind == "bell":
            response *= _peaking_response(freqs, b.freq_hz, b.gain_db, b.q)
        else:
            response *= _shelf_response(freqs, b.freq_hz, b.gain_db, b.kind)

    freqs_norm = freqs / (sr / 2)
    # frequency sampling: build via inverse FFT of desired response
    full_resp = np.interp(
        np.linspace(0, 1, numtaps // 2 + 1), freqs_norm, response
    )
    fir = np.fft.irfft(full_resp, n=numtaps)
    fir = np.fft.fftshift(fir)
    window = np.kaiser(numtaps, beta=8.0)
    fir = fir * window
    fir = fir / np.sum(fir) * np.sum(fir)  # keep as-is; DC normalized implicitly by design
    return fir


def apply_linear_phase_eq(y: np.ndarray, bands: List[EQBand], sr: int) -> np.ndarray:
    if not bands:
        return y
    fir = design_linear_phase_eq(bands, sr)
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        convolved = fftconvolve(y[ch], fir, mode="same")
        out[ch] = convolved
    return out


# ---------------------------------------------------------------------------
# Dynamic EQ (surgical, negative-only gain reduction at target frequencies)
# ---------------------------------------------------------------------------

def _bandpass_energy_envelope(mono, sr, f0, q, attack_ms, release_ms):
    """Extract an envelope of energy around f0 using a simple resonant
    bandpass (biquad-free, FFT-domain narrowband filter) then smooth it
    with one-pole attack/release."""
    n = len(mono)
    n_fft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(mono, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    bw = f0 / max(q, 0.1)
    mask = np.exp(-0.5 * ((freqs - f0) / max(bw, 1.0)) ** 2)
    band_signal = np.fft.irfft(spec * mask, n=n_fft)[:n]

    rectified = np.abs(band_signal)
    attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * release_ms / 1000.0))
    env = np.zeros(n)
    prev = 0.0
    for i in range(n):
        x = rectified[i]
        coeff = attack_coeff if x > prev else release_coeff
        prev = coeff * prev + (1 - coeff) * x
        env[i] = prev
    return env


def apply_dynamic_eq(y: np.ndarray, bands: List[DynamicEQBand], sr: int) -> np.ndarray:
    if not bands:
        return y
    out = y.copy()
    mono = np.mean(y, axis=0)
    for b in bands:
        env = _bandpass_energy_envelope(mono, sr, b.freq_hz, b.q, b.attack_ms, b.release_ms)
        env_db = 20 * np.log10(np.maximum(env, 1e-9))
        over = np.maximum(0.0, env_db - b.threshold_db)
        reduction_db = -np.minimum(over * (1 - 1 / max(b.ratio, 1.0)), abs(b.max_cut_db))
        gain_lin = 10 ** (reduction_db / 20.0)

        n = len(mono)
        n_fft = 1 << (n - 1).bit_length()
        freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
        bw = b.freq_hz / max(b.q, 0.1)
        mask = np.exp(-0.5 * ((freqs - b.freq_hz) / max(bw, 1.0)) ** 2)

        for ch in range(out.shape[0]):
            spec = np.fft.rfft(out[ch], n=n_fft)
            band_component = np.fft.irfft(spec * mask, n=n_fft)[:n]
            residual = out[ch] - band_component
            out[ch] = residual + band_component * gain_lin
    return out


# ---------------------------------------------------------------------------
# VCA-style bus compressor (feed-forward RMS detector, soft knee)
# ---------------------------------------------------------------------------

def apply_compressor(y: np.ndarray, params: CompressorParams, sr: int) -> np.ndarray:
    mono = np.mean(y, axis=0)
    n = len(mono)

    window_len = max(1, int(sr * 0.003))
    kernel = np.ones(window_len) / window_len
    sq = mono ** 2
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-12)
    level_db = 20 * np.log10(np.maximum(rms, 1e-9))

    knee_db = 3.0
    over = level_db - params.threshold_db
    gain_reduction_db = np.zeros(n)
    hard_over = over > knee_db / 2
    soft_zone = np.abs(over) <= knee_db / 2
    gain_reduction_db[hard_over] = -(over[hard_over] * (1 - 1 / params.ratio))
    if np.any(soft_zone):
        x = over[soft_zone] + knee_db / 2
        gain_reduction_db[soft_zone] = -((x ** 2) / (2 * knee_db)) * (1 - 1 / params.ratio)

    attack_coeff = np.exp(-1.0 / (sr * params.attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * params.release_ms / 1000.0))
    smoothed = np.zeros(n)
    prev = 0.0
    for i in range(n):
        target = gain_reduction_db[i]
        coeff = attack_coeff if target < prev else release_coeff
        prev = coeff * prev + (1 - coeff) * target
        smoothed[i] = prev

    gain_lin = 10 ** ((smoothed + params.makeup_db) / 20.0)
    compressed = y * gain_lin[np.newaxis, :]
    return y * (1 - params.mix) + compressed * params.mix


# ---------------------------------------------------------------------------
# Harmonic saturation
# ---------------------------------------------------------------------------

def apply_saturation(y: np.ndarray, params: SaturationParams) -> np.ndarray:
    if params.mix <= 0.0:
        return y
    drive_lin = 10 ** (params.drive_db / 20.0)
    driven = y * drive_lin
    if params.kind == "tanh":
        saturated = np.tanh(driven)
    else:
        saturated = np.clip(driven, -1.0, 1.0)
    peak = np.max(np.abs(saturated)) + 1e-9
    if peak > 1e-6:
        saturated = saturated / peak * np.max(np.abs(y) + 1e-9)
    return y * (1 - params.mix) + saturated * params.mix


# ---------------------------------------------------------------------------
# Stereo width control (mid/side)
# ---------------------------------------------------------------------------

def apply_stereo_width(y: np.ndarray, mult: float) -> np.ndarray:
    if y.shape[0] < 2 or abs(mult - 1.0) < 1e-6:
        return y
    left, right = y[0], y[1]
    mid = (left + right) / 2
    side = (left - right) / 2 * mult
    new_left = mid + side
    new_right = mid - side
    return np.stack([new_left, new_right])


# ---------------------------------------------------------------------------
# True-peak-aware brickwall limiter (lookahead, oversampled peak detection)
# ---------------------------------------------------------------------------

def apply_limiter(y: np.ndarray, params: LimiterParams, sr: int, oversample: int = 4) -> np.ndarray:
    ceiling_lin = 10 ** (params.ceiling_dbtp / 20.0)
    lookahead_samples = max(1, int(sr * params.lookahead_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (sr * params.release_ms / 1000.0))

    up = np.stack([resample_poly(ch, oversample, 1) for ch in y])
    peak_env = np.max(np.abs(up), axis=0)

    n_os = up.shape[1]
    look = lookahead_samples * oversample

    from scipy.ndimage import maximum_filter1d
    future_max = maximum_filter1d(peak_env, size=look * 2 + 1, mode="nearest")

    target_gain = np.minimum(1.0, ceiling_lin / np.maximum(future_max, 1e-9))

    smoothed_gain = np.ones(n_os)
    prev = 1.0
    for i in range(n_os):
        t = target_gain[i]
        if t < prev:
            prev = t  # instant attack to prevent overshoot (true peak safety)
        else:
            prev = release_coeff * prev + (1 - release_coeff) * t
        smoothed_gain[i] = prev

    limited_up = up * smoothed_gain[np.newaxis, :]
    limited = np.stack([resample_poly(ch, 1, oversample) for ch in limited_up])
    limited = limited[:, : y.shape[1]]

    # Safety net #1: raw sample-peak trim (cheap, catches gross overshoot).
    final_peak = np.max(np.abs(limited))
    if final_peak > ceiling_lin:
        limited = limited * (ceiling_lin / final_peak)

    # Safety net #2: the down/up-sample round-trip through resample_poly can
    # introduce small reconstruction ripple that nudges the TRUE peak
    # (inter-sample peak) slightly above what the sample-domain check sees.
    # Re-check true peak via a fresh oversampled pass and trim again if
    # needed -- true-peak compliance must never be a "best effort".
    reup = np.stack([resample_poly(ch, oversample, 1) for ch in limited])
    true_peak_lin = float(np.max(np.abs(reup)))
    if true_peak_lin > ceiling_lin:
        limited = limited * (ceiling_lin / true_peak_lin) * 0.999  # tiny extra margin

    return limited


# ---------------------------------------------------------------------------
# Full chain
# ---------------------------------------------------------------------------

def render_chain(y: np.ndarray, sr: int, params: ChainParams) -> np.ndarray:
    """Run the full serial mastering chain and return the processed buffer."""
    out = y * (10 ** (params.input_trim_db / 20.0))
    out = apply_linear_phase_eq(out, params.linear_eq, sr)
    out = apply_dynamic_eq(out, params.dynamic_eq, sr)
    out = apply_compressor(out, params.compressor, sr)
    out = apply_saturation(out, params.saturation)
    out = apply_stereo_width(out, params.stereo_width_mult)
    out = apply_limiter(out, params.limiter, sr)
    return out
