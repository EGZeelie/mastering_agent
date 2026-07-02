"""Tests for the Analysis Engine (the 'Ears')."""
import numpy as np
import pytest

from mastering_agent import analysis


def test_analyze_returns_populated_features(flawed_mix_path):
    feats = analysis.analyze(flawed_mix_path)
    assert feats.sample_rate == analysis.SAMPLE_RATE_ANALYSIS
    assert feats.duration_sec == pytest.approx(3.0, abs=0.05)
    assert isinstance(feats.lufs_integrated, float)
    assert feats.true_peak_dbtp < 6.0  # sanity: not a runaway value
    assert feats.crest_factor_db > 0
    assert set(feats.band_energy_db.keys()) == set(analysis.BANDS.keys())


def test_problem_detector_flags_known_flaws(flawed_mix_path):
    """The synthetic flawed mix has deliberately injected sub-bass buildup
    and a harsh ~3.2kHz resonance -- the detector must catch both."""
    feats = analysis.analyze(flawed_mix_path)
    kinds = {p.kind for p in feats.problems}
    assert "sub_bass_buildup" in kinds
    assert "harsh_resonance" in kinds


def test_problem_detector_silent_on_clean_reference(reference_master_path):
    """The reference master is tonally balanced -- it should NOT trip the
    same sub-bass/harshness flags as the deliberately-flawed mix (it may
    still trip 'already_hot'/'over_compressed' since it's mastered)."""
    feats = analysis.analyze(reference_master_path)
    kinds = {p.kind for p in feats.problems}
    assert "sub_bass_buildup" not in kinds
    assert "harsh_resonance" not in kinds


def test_band_energy_db_is_referenced_not_absolute(flawed_mix_path):
    """Band energies should be small relative dB values (referenced to the
    track's own average), not raw unbounded STFT power in dB."""
    feats = analysis.analyze(flawed_mix_path)
    for band, val in feats.band_energy_db.items():
        assert -60.0 < val < 60.0, f"{band} band energy {val} looks unreferenced"


def test_genre_profiling_returns_valid_profile(flawed_mix_path):
    feats = analysis.analyze(flawed_mix_path)
    profile = analysis.profile_genre(feats)
    assert profile["matched_genre"] in analysis.GENRE_PROFILES
    assert "target_lufs" in profile
    assert "min_crest_db" in profile
    assert profile["distance"] >= 0


def test_reference_diff_signs_are_sensible(flawed_mix_path, reference_master_path):
    mix_feats = analysis.analyze(flawed_mix_path)
    ref_feats = analysis.analyze(reference_master_path)
    diff = analysis.diff_against_reference(mix_feats, ref_feats)

    # The reference master is deliberately built louder than the flawed mix,
    # so the loudness delta the agent needs to apply must be positive.
    assert diff["loudness_delta_lufs"] > 0
    assert diff["reference_lufs"] == pytest.approx(ref_feats.lufs_integrated)
    assert set(diff["band_delta_db"].keys()) == set(analysis.BANDS.keys())


def test_analyze_handles_near_silence_without_crashing(silence_path):
    """Edge case: near-silent input must not raise (e.g. log(0) crashes)."""
    feats = analysis.analyze(silence_path)
    assert feats.lufs_integrated < -40.0  # should register as very quiet
    assert not np.isnan(feats.crest_factor_db)
    assert not np.isnan(feats.true_peak_dbtp)
