"""Tests for the Agentic Decision Core (the 'Brain'), especially the
hard safety constraints -- these are the most important tests in the suite,
since they guard the project's core differentiator (never over-limit)."""
import copy


from mastering_agent import analysis, decision, dsp


def test_rule_based_engine_produces_valid_chain(flawed_mix_path):
    feats = analysis.analyze(flawed_mix_path)
    profile = analysis.profile_genre(feats)
    engine = decision.RuleBasedDecisionEngine()
    params = engine.propose_chain_update(feats, profile, None, iteration=0)
    assert isinstance(params, dsp.ChainParams)
    assert params.limiter.ceiling_dbtp <= -0.8


def test_rule_based_engine_addresses_detected_problems(flawed_mix_path):
    """If sub-bass buildup and harsh resonance are detected, the proposed
    chain should contain corrective EQ/dynamic-EQ addressing them."""
    feats = analysis.analyze(flawed_mix_path)
    profile = analysis.profile_genre(feats)
    engine = decision.RuleBasedDecisionEngine()
    params = engine.propose_chain_update(feats, profile, None, iteration=0)

    kinds = {p.kind for p in feats.problems}
    if "sub_bass_buildup" in kinds:
        assert any(b.kind == "low_shelf" and b.gain_db < 0 for b in params.linear_eq)
    if "harsh_resonance" in kinds:
        assert len(params.dynamic_eq) > 0
        assert all(b.max_cut_db > 0 for b in params.dynamic_eq)  # cut magnitude always positive


def test_gemini_engine_falls_back_without_api_key(flawed_mix_path, monkeypatch):
    """Without GEMINI_API_KEY/GOOGLE_API_KEY set, the engine must not crash
    or hang -- it must transparently use the rule-based fallback."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    engine = decision.GeminiDecisionEngine()
    assert engine._client is None

    feats = analysis.analyze(flawed_mix_path)
    profile = analysis.profile_genre(feats)
    params = engine.propose_chain_update(feats, profile, None, iteration=0)

    assert isinstance(params, dsp.ChainParams)
    assert engine.used_fallback_last_call is True
    assert "no_api_key" in engine.name or "fallback" in (engine.last_reasoning or "").lower()


def test_gemini_engine_falls_back_on_call_failure(flawed_mix_path, monkeypatch):
    """Even if a client IS configured, any exception during the actual model
    call must be caught and routed to the fallback -- never propagate and
    stall the mastering loop."""
    engine = decision.GeminiDecisionEngine()

    class BoomClient:
        pass

    engine._client = BoomClient()
    engine._client_kind = "genai"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(engine, "_call_model", boom)

    feats = analysis.analyze(flawed_mix_path)
    profile = analysis.profile_genre(feats)
    params = engine.propose_chain_update(feats, profile, None, iteration=0)

    assert isinstance(params, dsp.ChainParams)
    assert engine.used_fallback_last_call is True
    assert "simulated network failure" in engine.last_reasoning


# ---------------------------------------------------------------------------
# Hard constraint enforcement -- the project's core safety guarantee
# ---------------------------------------------------------------------------

def test_hard_constraints_clamp_true_peak_ceiling(flawed_mix_path):
    feats = analysis.analyze(flawed_mix_path)
    bad_params = dsp.ChainParams(
        limiter=dsp.LimiterParams(ceiling_dbtp=-0.1, release_ms=30, lookahead_ms=2),
    )
    fixed = decision.enforce_hard_constraints(bad_params, feats)
    assert fixed.limiter.ceiling_dbtp <= decision.TRUE_PEAK_CEILING_DBTP


def test_hard_constraints_clamp_extreme_compressor_ratio(flawed_mix_path):
    feats = analysis.analyze(flawed_mix_path)
    bad_params = dsp.ChainParams(
        compressor=dsp.CompressorParams(threshold_db=-30, ratio=20.0, attack_ms=1, release_ms=30, makeup_db=10, mix=1.0),
    )
    fixed = decision.enforce_hard_constraints(bad_params, feats)
    assert fixed.compressor.ratio <= 4.0
    assert fixed.compressor.mix <= 0.9


def test_hard_constraints_never_let_projected_crest_fall_below_floor(flawed_mix_path):
    """This is the single most important test in the suite: no matter how
    aggressive an upstream proposal (LLM or otherwise) is, the projected
    crest factor after enforcement must respect the hard floor."""
    feats = analysis.analyze(flawed_mix_path)

    for ratio in [2.0, 4.0, 8.0, 15.0, 30.0]:
        for mix in [0.5, 0.8, 1.0]:
            bad_params = dsp.ChainParams(
                compressor=dsp.CompressorParams(
                    threshold_db=-30, ratio=ratio, attack_ms=1, release_ms=30, makeup_db=10, mix=mix,
                ),
            )
            fixed = decision.enforce_hard_constraints(copy.deepcopy(bad_params), feats)
            projected_crest_loss = (fixed.compressor.ratio - 1.0) * fixed.compressor.mix * 1.4
            projected_crest = feats.crest_factor_db - projected_crest_loss
            assert projected_crest >= decision.MIN_CREST_FACTOR_DB - 0.01, (
                f"ratio={ratio} mix={mix} -> projected crest {projected_crest} "
                f"violates floor {decision.MIN_CREST_FACTOR_DB}"
            )


def test_hard_constraints_ease_off_more_when_mix_already_near_floor(flawed_mix_path):
    """If the incoming mix already has a low crest factor (pre-squashed),
    the governor should be even more conservative than for a dynamic mix."""
    import dataclasses

    feats = analysis.analyze(flawed_mix_path)
    near_floor_feats = dataclasses.replace(feats, crest_factor_db=decision.MIN_CREST_FACTOR_DB + 1.0)

    params_a = dsp.ChainParams(
        compressor=dsp.CompressorParams(threshold_db=-30, ratio=4.0, attack_ms=1, release_ms=30, makeup_db=5, mix=0.9),
    )
    params_b = copy.deepcopy(params_a)

    fixed_normal = decision.enforce_hard_constraints(params_a, feats)
    fixed_near_floor = decision.enforce_hard_constraints(params_b, near_floor_feats)

    assert fixed_near_floor.compressor.ratio <= fixed_normal.compressor.ratio
    assert fixed_near_floor.compressor.mix <= fixed_normal.compressor.mix
