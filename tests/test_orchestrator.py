"""End-to-end tests for the iterative control loop. Kept fast (short audio,
few iterations) so this can run in CI on every push without becoming a
bottleneck."""
import os

from mastering_agent import decision, orchestrator


def test_full_loop_converges_genre_profile_mode(flawed_mix_path, tmp_path):
    output_path = str(tmp_path / "mastered_genre.wav")
    result = orchestrator.run_mastering_loop(
        input_path=flawed_mix_path,
        output_path=output_path,
        reference_path=None,
        engine=decision.RuleBasedDecisionEngine(),
        max_iterations=6,
        verbose=False,
    )
    assert os.path.exists(output_path)
    assert result.final_features["crest_factor_db"] >= decision.MIN_CREST_FACTOR_DB - 0.5
    assert result.final_features["true_peak_dbtp"] <= orchestrator.TRUE_PEAK_HARD_CEILING + 0.1


def test_full_loop_converges_reference_match_mode(flawed_mix_path, reference_master_path, tmp_path):
    output_path = str(tmp_path / "mastered_ref.wav")
    result = orchestrator.run_mastering_loop(
        input_path=flawed_mix_path,
        output_path=output_path,
        reference_path=reference_master_path,
        engine=decision.RuleBasedDecisionEngine(),
        max_iterations=6,
        verbose=False,
    )
    assert os.path.exists(output_path)
    assert result.final_features["crest_factor_db"] >= decision.MIN_CREST_FACTOR_DB - 0.5
    assert result.final_features["true_peak_dbtp"] <= orchestrator.TRUE_PEAK_HARD_CEILING + 0.1


def test_output_file_is_valid_audio_no_nans(flawed_mix_path, tmp_path):
    import numpy as np
    import soundfile as sf

    output_path = str(tmp_path / "mastered.wav")
    orchestrator.run_mastering_loop(
        input_path=flawed_mix_path,
        output_path=output_path,
        engine=decision.RuleBasedDecisionEngine(),
        max_iterations=4,
        verbose=False,
    )
    y, sr = sf.read(output_path)
    assert not np.any(np.isnan(y))
    assert np.max(np.abs(y)) <= 1.0 + 1e-6


def test_report_serializes_to_json(flawed_mix_path, tmp_path):
    output_path = str(tmp_path / "mastered.wav")
    report_path = str(tmp_path / "report.json")
    result = orchestrator.run_mastering_loop(
        input_path=flawed_mix_path,
        output_path=output_path,
        engine=decision.RuleBasedDecisionEngine(),
        max_iterations=3,
        verbose=False,
    )
    orchestrator.save_report(result, report_path)
    assert os.path.exists(report_path)

    import json
    with open(report_path) as f:
        data = json.load(f)
    assert "iterations" in data
    assert "final_features" in data
    assert data["engine_used"] == "rule_based"


def test_engine_used_reports_fallback_transparently(flawed_mix_path, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    output_path = str(tmp_path / "mastered.wav")
    result = orchestrator.run_mastering_loop(
        input_path=flawed_mix_path,
        output_path=output_path,
        engine=decision.GeminiDecisionEngine(),
        max_iterations=3,
        verbose=False,
    )
    assert "fallback" in result.engine_used.lower() or "no_api_key" in result.engine_used
