"""
The Analysis Engine ("The Ears")
=================================

Extracts objective features from a mixdown so the decision core has something
real to reason about, instead of guessing. Everything here is measured, not
learned-from-nowhere: LUFS integrated loudness (ITU-R BS.1770 via pyloudnorm),
true-peak estimate (4x oversampled), crest factor, stereo correlation/width,
1/3-octave-ish band energy for tonal balance, spectral centroid/tilt, and a
lightweight "problem detector" for sub-bass buildup and harsh upper-mid
resonances.

Also implements:
  - Genre profiling: nearest-match against a small bank of target tonal/
    loudness curves (a stand-in for EMA-style unsupervised clustering --
    swap `GENRE_PROFILES` for a trained k-means/GMM model without changing
    the calling contract).
  - Reference matching: computes the *differential* between a mix and a
    reference master across loudness, tone (band-by-band dB deltas), and
    stereo width, which the decision core turns into corrective deltas.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import librosa
import pyloudnorm as pyln
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Band definitions used throughout the engine (Hz). Coarser than 1/3-octave
# but detailed enough to drive real EQ decisions without being noisy.
# ---------------------------------------------------------------------------
BANDS = {
    "sub_bass":     (20, 60),
    "bass":         (60, 250),
    "low_mid":      (250, 500),
    "mid":          (500, 2000),
    "high_mid":     (2000, 4000),
    "presence":     (4000, 6000),
    "brilliance":   (6000, 16000),
    "air":          (16000, 20000),
}

SAMPLE_RATE_ANALYSIS = 48000


def _to_jsonable(obj):
    """Recursively coerce numpy scalar/array types to native Python types
    so the result is safe to pass to json.dumps()."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class ProblemFlag:
    kind: str            # e.g. "sub_bass_buildup", "harsh_resonance"
    band_hz: tuple
    severity: float       # 0..1, higher = worse
    detail: str


@dataclass
class AudioFeatures:
    sample_rate: int
    duration_sec: float
    lufs_integrated: float
    true_peak_dbtp: float
    crest_factor_db: float
    rms_dbfs: float
    stereo_width: float          # 0 (mono) .. ~1.5 (very wide/decorrelated)
    stereo_correlation: float    # -1..1
    band_energy_db: dict         # band name -> relative dB level
    spectral_centroid_hz: float
    spectral_tilt_db_per_oct: float
    tempo_bpm: Optional[float]
    problems: list = field(default_factory=list)

    def to_dict(self):
        """Return a plain-JSON-serializable dict. Numpy scalar types (e.g.
        np.float32/np.float64) can leak into AudioFeatures from upstream
        numpy computations, so every value is explicitly coerced here --
        without this, json.dumps() (used both by save_report() and by
        GeminiDecisionEngine's prompt builder) raises TypeError on the
        first non-native-Python numeric field it hits."""
        d = dataclasses.asdict(self)
        d["problems"] = [dataclasses.asdict(p) for p in self.problems]
        return _to_jsonable(d)


def _load_stereo(path: str, sr: int = SAMPLE_RATE_ANALYSIS) -> np.ndarray:
    """Load audio as shape (2, n) float32, always stereo (mono is duplicated)."""
    y, orig_sr = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def _true_peak_dbtp(y: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Estimate true peak via oversampling (poly-phase), per ITU-R BS.1770 spirit."""
    peak = 0.0
    for ch in y:
        up = resample_poly(ch, oversample, 1)
        peak = max(peak, float(np.max(np.abs(up))))
    peak = max(peak, 1e-9)
    return 20 * np.log10(peak)


def _crest_factor_db(y: np.ndarray) -> float:
    mono = np.mean(y, axis=0)
    peak = np.max(np.abs(mono)) + 1e-12
    rms = np.sqrt(np.mean(mono ** 2)) + 1e-12
    return 20 * np.log10(peak / rms)


def _rms_dbfs(y: np.ndarray) -> float:
    mono = np.mean(y, axis=0)
    rms = np.sqrt(np.mean(mono ** 2)) + 1e-12
    return 20 * np.log10(rms)


def _stereo_metrics(y: np.ndarray) -> tuple:
    if y.shape[0] < 2:
        return 0.0, 1.0
    left, right = y[0], y[1]
    if np.std(left) < 1e-9 or np.std(right) < 1e-9:
        corr = 1.0
    else:
        corr = float(np.corrcoef(left, right)[0, 1])
    mid = (left + right) / 2
    side = (left - right) / 2
    mid_rms = np.sqrt(np.mean(mid ** 2)) + 1e-12
    side_rms = np.sqrt(np.mean(side ** 2)) + 1e-12
    width = float(side_rms / mid_rms)
    return width, corr


def _band_energy_db(y: np.ndarray, sr: int) -> dict:
    """Per-band energy in dB, referenced to the track's own broadband level.

    This makes values track-agnostic (a value of 0 dB means "as loud as the
    track overall", negative means quieter than average) so they read
    sensibly in reports AND so cross-track comparisons (reference matching)
    reflect tonal *shape* differences rather than being dominated by one
    track simply being louder overall.
    """
    mono = np.mean(y, axis=0)
    n_fft = 8192
    hop = 2048
    S = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power_spec = np.mean(S, axis=1)

    raw_band_db = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            raw_band_db[name] = -100.0
            continue
        band_power = np.mean(power_spec[mask]) + 1e-15
        raw_band_db[name] = float(10 * np.log10(band_power))

    # Reference against the mean of the per-band values (equal per-octave
    # weighting) rather than a raw linear-FFT-bin average, which would be
    # dominated by bin count in the high-frequency region and produce
    # misleadingly large numbers.
    broadband_db = float(np.mean(list(raw_band_db.values())))
    return {name: val - broadband_db for name, val in raw_band_db.items()}


def _spectral_centroid_and_tilt(y: np.ndarray, sr: int) -> tuple:
    mono = np.mean(y, axis=0)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))
    n_fft = 8192
    S = np.abs(librosa.stft(mono, n_fft=n_fft)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = np.mean(S, axis=1)
    valid = freqs > 20
    log_f = np.log2(freqs[valid])
    log_p_db = 10 * np.log10(power[valid] + 1e-15)
    tilt = float(np.polyfit(log_f, log_p_db, 1)[0])  # dB per octave
    return centroid, tilt


def _detect_problems(band_db: dict, features_partial: dict) -> list:
    problems = []

    sub = band_db["sub_bass"]
    bass = band_db["bass"]
    if sub - bass > 3.0:
        problems.append(ProblemFlag(
            kind="sub_bass_buildup",
            band_hz=BANDS["sub_bass"],
            severity=min(1.0, (sub - bass) / 12.0),
            detail=f"Sub-bass ({sub:.1f} dB) exceeds bass band ({bass:.1f} dB) by "
                   f"{sub - bass:.1f} dB -- likely uncontrolled low-end buildup / room mud.",
        ))

    high_mid = band_db["high_mid"]
    presence = band_db["presence"]
    mid = band_db["mid"]
    avg_neighbors = (mid + band_db["brilliance"]) / 2
    if presence - avg_neighbors > 4.0 or high_mid - avg_neighbors > 4.0:
        worst_band = "presence" if (presence - avg_neighbors) > (high_mid - avg_neighbors) else "high_mid"
        problems.append(ProblemFlag(
            kind="harsh_resonance",
            band_hz=BANDS[worst_band],
            severity=min(1.0, max(presence, high_mid) - avg_neighbors) / 10.0,
            detail=f"Upper-mid/presence energy spikes {max(presence - avg_neighbors, high_mid - avg_neighbors):.1f} dB "
                   f"above neighboring bands -- likely harsh vocal/synth resonance around "
                   f"{BANDS[worst_band][0]}-{BANDS[worst_band][1]} Hz.",
        ))

    crest = features_partial.get("crest_factor_db", 12.0)
    if crest < 6.0:
        problems.append(ProblemFlag(
            kind="over_compressed",
            band_hz=(20, 20000),
            severity=min(1.0, (6.0 - crest) / 6.0),
            detail=f"Crest factor already low ({crest:.1f} dB) -- mix is pre-squashed; "
                   f"mastering chain must apply minimal further compression/limiting.",
        ))

    lufs = features_partial.get("lufs_integrated", -14.0)
    if lufs > -8.0:
        problems.append(ProblemFlag(
            kind="already_hot",
            band_hz=(20, 20000),
            severity=min(1.0, (lufs + 8.0) / 6.0),
            detail=f"Mix is already very loud ({lufs:.1f} LUFS) going into mastering -- "
                   f"limiter will have little headroom to work with.",
        ))

    return problems


def analyze(path: str) -> AudioFeatures:
    """Run the full analysis engine on an audio file and return AudioFeatures."""
    y = _load_stereo(path)
    sr = SAMPLE_RATE_ANALYSIS
    duration = y.shape[1] / sr

    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y.T))

    true_peak = _true_peak_dbtp(y, sr)
    crest = _crest_factor_db(y)
    rms = _rms_dbfs(y)
    width, corr = _stereo_metrics(y)
    band_db = _band_energy_db(y, sr)
    centroid, tilt = _spectral_centroid_and_tilt(y, sr)

    tempo = None
    try:
        mono = np.mean(y, axis=0)
        tempo_arr, _ = librosa.beat.beat_track(y=mono, sr=sr)
        tempo_arr = np.atleast_1d(tempo_arr)
        tempo = float(np.mean(tempo_arr)) if tempo_arr.size else None
        if tempo <= 0:
            tempo = None
    except Exception:
        tempo = None

    partial = {"crest_factor_db": crest, "lufs_integrated": lufs}
    problems = _detect_problems(band_db, partial)

    return AudioFeatures(
        sample_rate=sr,
        duration_sec=duration,
        lufs_integrated=lufs,
        true_peak_dbtp=true_peak,
        crest_factor_db=crest,
        rms_dbfs=rms,
        stereo_width=width,
        stereo_correlation=corr,
        band_energy_db=band_db,
        spectral_centroid_hz=centroid,
        spectral_tilt_db_per_oct=tilt,
        tempo_bpm=tempo,
        problems=problems,
    )


# ---------------------------------------------------------------------------
# Genre profiling
# ---------------------------------------------------------------------------
# Stand-in for an unsupervised clustering model (EMA-style). Each profile
# encodes a *target* tonal curve (relative band dB, roughly bass-referenced)
# and target loudness/crest-factor envelope for that style. In production
# this bank would be learned via k-means/GMM over a large corpus of
# commercial masters; the calling contract (nearest-centroid match) is
# identical either way, so swapping in a trained model is a drop-in change.

GENRE_PROFILES = {
    "aggressive_synth_pop": {
        "target_band_db": {
            "sub_bass": -6.0, "bass": -2.0, "low_mid": -4.0, "mid": -3.0,
            "high_mid": -2.5, "presence": -2.0, "brilliance": -3.5, "air": -8.0,
        },
        "target_lufs": -8.0,
        "min_crest_db": 7.0,
        "target_width": 1.05,
    },
    "contemporary_rock": {
        "target_band_db": {
            "sub_bass": -10.0, "bass": -3.5, "low_mid": -3.0, "mid": -2.0,
            "high_mid": -3.0, "presence": -3.5, "brilliance": -5.5, "air": -11.0,
        },
        "target_lufs": -9.5,
        "min_crest_db": 8.5,
        "target_width": 0.9,
    },
    "acoustic_singer_songwriter": {
        "target_band_db": {
            "sub_bass": -16.0, "bass": -6.0, "low_mid": -3.5, "mid": -2.0,
            "high_mid": -3.5, "presence": -4.5, "brilliance": -7.0, "air": -13.0,
        },
        "target_lufs": -13.0,
        "min_crest_db": 11.0,
        "target_width": 0.75,
    },
    "electronic_dance": {
        "target_band_db": {
            "sub_bass": -2.0, "bass": -1.0, "low_mid": -5.0, "mid": -3.5,
            "high_mid": -3.0, "presence": -2.5, "brilliance": -4.0, "air": -9.0,
        },
        "target_lufs": -7.0,
        "min_crest_db": 6.5,
        "target_width": 1.15,
    },
}


def _normalize_curve(band_db: dict) -> np.ndarray:
    vec = np.array([band_db[b] for b in BANDS])
    return vec - vec[list(BANDS).index("bass")]  # normalize relative to bass


def profile_genre(features: AudioFeatures) -> dict:
    """Match measured tonal balance against the genre profile bank (nearest centroid)."""
    mix_curve = _normalize_curve(features.band_energy_db)
    best_name, best_dist = None, float("inf")
    for name, prof in GENRE_PROFILES.items():
        target_curve = _normalize_curve(prof["target_band_db"])
        dist = float(np.linalg.norm(mix_curve - target_curve))
        if dist < best_dist:
            best_name, best_dist = name, dist
    profile = GENRE_PROFILES[best_name]
    return {
        "matched_genre": best_name,
        "distance": best_dist,
        "target_band_db": profile["target_band_db"],
        "target_lufs": profile["target_lufs"],
        "min_crest_db": profile["min_crest_db"],
        "target_width": profile["target_width"],
    }


# ---------------------------------------------------------------------------
# Reference matching
# ---------------------------------------------------------------------------

def diff_against_reference(mix: AudioFeatures, reference: AudioFeatures) -> dict:
    """Compute the differential the decision core needs to bridge mix -> reference."""
    band_deltas = {
        b: reference.band_energy_db[b] - mix.band_energy_db[b] for b in BANDS
    }
    return {
        "loudness_delta_lufs": reference.lufs_integrated - mix.lufs_integrated,
        "width_delta": reference.stereo_width - mix.stereo_width,
        "crest_delta_db": reference.crest_factor_db - mix.crest_factor_db,
        "tilt_delta_db_per_oct": reference.spectral_tilt_db_per_oct - mix.spectral_tilt_db_per_oct,
        "band_delta_db": band_deltas,
        "reference_lufs": reference.lufs_integrated,
        "reference_crest_db": reference.crest_factor_db,
        "reference_width": reference.stereo_width,
    }
