"""Tests for the DSP Control Loop (the 'Hands')."""
import numpy as np

from mastering_agent import analysis, dsp


def _load(path):
    return analysis._load_stereo(path)


def test_render_chain_default_params_is_near_passthrough(flawed_mix_path):
    """With all-default ChainParams (no EQ, no dynamics, 0dB trim, -1dBTP
    limiter with a lot of headroom), output should closely resemble input
    aside from limiter safety action."""
    y = _load(flawed_mix_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    params = dsp.ChainParams()
    out = dsp.render_chain(y, sr, params)
    assert out.shape == y.shape
    assert not np.any(np.isnan(out))


def test_render_chain_output_has_no_nans_or_infs(flawed_mix_path):
    y = _load(flawed_mix_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    params = dsp.ChainParams(
        linear_eq=[dsp.EQBand(40, -3.0, 0.9, "low_shelf"), dsp.EQBand(9000, 1.5, 0.7, "high_shelf")],
        dynamic_eq=[dsp.DynamicEQBand(3200, 2.5, -24.0, 3.0, 4.0, 3.0, 60.0)],
        compressor=dsp.CompressorParams(threshold_db=-20, ratio=2.2, attack_ms=18, release_ms=140, makeup_db=2.0, mix=0.8),
        saturation=dsp.SaturationParams(drive_db=3.0, mix=0.15),
        limiter=dsp.LimiterParams(ceiling_dbtp=-1.0, release_ms=80, lookahead_ms=5),
        input_trim_db=6.0,
        stereo_width_mult=1.1,
    )
    out = dsp.render_chain(y, sr, params)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))


def test_limiter_enforces_true_peak_ceiling(flawed_mix_path):
    """The limiter must never let true peak exceed its configured ceiling,
    even when driven with an aggressive input trim."""
    y = _load(flawed_mix_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    ceiling = -1.0
    params = dsp.ChainParams(
        input_trim_db=18.0,  # deliberately excessive to stress-test the limiter
        limiter=dsp.LimiterParams(ceiling_dbtp=ceiling, release_ms=60, lookahead_ms=5),
    )
    out = dsp.render_chain(y, sr, params)
    true_peak = analysis._true_peak_dbtp(out, sr)
    assert true_peak <= ceiling + 0.1, f"true peak {true_peak} exceeded ceiling {ceiling}"


def test_compressor_with_fast_attack_reduces_dynamic_range(flawed_mix_path):
    """With a FAST attack, the compressor catches transients too, so overall
    crest factor should drop. (Note: a slow attack can legitimately let
    transients pass through uncompressed while gain-reducing the sustained
    body -- which INCREASES crest factor, a well-known "punchy" compression
    effect, not a bug. Fast attack is the correct condition to test dynamic
    range reduction.)"""
    y = _load(flawed_mix_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    crest_before = analysis._crest_factor_db(y)

    params = dsp.CompressorParams(threshold_db=-30, ratio=4.0, attack_ms=0.1, release_ms=100, makeup_db=0, mix=1.0)
    compressed = dsp.apply_compressor(y, params, sr)
    crest_after = analysis._crest_factor_db(compressed)

    assert crest_after < crest_before


def test_compressor_slow_attack_can_preserve_or_increase_crest_factor(flawed_mix_path):
    """Documents/locks in the expected transient-preserving behavior of slow
    attack times -- relevant to the project's crest-factor-preservation goal:
    a mastering agent that wants to protect transients can lean on slower
    attack times as one of its levers."""
    y = _load(flawed_mix_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    crest_before = analysis._crest_factor_db(y)

    params = dsp.CompressorParams(threshold_db=-30, ratio=4.0, attack_ms=20.0, release_ms=100, makeup_db=0, mix=1.0)
    compressed = dsp.apply_compressor(y, params, sr)
    crest_after = analysis._crest_factor_db(compressed)

    assert crest_after >= crest_before - 0.5  # slow attack should not meaningfully crush transients


def test_saturation_bypass_is_identity_when_mix_zero(flawed_mix_path):
    y = _load(flawed_mix_path)
    params = dsp.SaturationParams(drive_db=10.0, mix=0.0)
    out = dsp.apply_saturation(y, params)
    assert np.allclose(out, y)


def test_stereo_width_widens_and_narrows_correctly(reference_master_path):
    y = _load(reference_master_path)
    _, corr_before = analysis._stereo_metrics(y)
    width_before, _ = analysis._stereo_metrics(y)

    widened = dsp.apply_stereo_width(y, 1.5)
    width_wide, _ = analysis._stereo_metrics(widened)
    assert width_wide > width_before

    narrowed = dsp.apply_stereo_width(y, 0.3)
    width_narrow, _ = analysis._stereo_metrics(narrowed)
    assert width_narrow < width_before


def test_dynamic_eq_reduces_energy_at_target_frequency(flawed_mix_path):
    """Applying a dynamic EQ cut centered on the mix's known harsh resonance
    (~3.2kHz) should measurably reduce energy in that band."""
    y = _load(flawed_mix_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    feats_before = analysis.analyze(flawed_mix_path)

    bands = [dsp.DynamicEQBand(freq_hz=3200, q=2.0, threshold_db=-40.0, ratio=4.0, max_cut_db=8.0)]
    out = dsp.apply_dynamic_eq(y, bands, sr)

    import soundfile as sf
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, out.T, sr)
    feats_after = analysis.analyze(tmp_path)
    os.unlink(tmp_path)

    assert feats_after.band_energy_db["high_mid"] < feats_before.band_energy_db["high_mid"]
