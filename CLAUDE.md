# CLAUDE.md

Guidance for Claude Code (or any agentic coding assistant) working in this
repository. Claude Code automatically reads this file for project context
when you run `claude` inside this directory — keep it accurate as the code
evolves.

## Project overview

An autonomous, closed-loop music mastering agent structured as three
decoupled layers plus an orchestrating loop:

- `mastering_agent/analysis.py` — **Ears**: feature extraction (LUFS, true
  peak, crest factor, stereo width, band energy, spectral tilt), genre
  profiling (nearest-centroid match), reference-track differential.
- `mastering_agent/decision.py` — **Brain**: `GeminiDecisionEngine` (LLM,
  structured JSON output) and `RuleBasedDecisionEngine` (deterministic
  fallback), plus `enforce_hard_constraints()` — the safety governor.
- `mastering_agent/dsp.py` — **Hands**: headless DSP chain (linear-phase EQ
  → dynamic EQ → VCA compressor → saturation → mid/side width → true-peak
  limiter), driven entirely by the `ChainParams` dataclass.
- `mastering_agent/orchestrator.py` — **Loop**: iterative
  analyze→decide→render→re-analyze cycle until convergence.

Full architecture rationale is in `README.md`; setup is in `INSTALL.md`.

## Setup for Claude Code sessions

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gemini]"        # or: pip install -r requirements.txt
python3 make_synthetic_mix.py      # regenerate test audio if missing
```

No `GEMINI_API_KEY` is required to run or test anything — the decision
engine transparently falls back to the deterministic rule-based engine.
Only set `GEMINI_API_KEY` / `GOOGLE_API_KEY` if you're specifically testing
the LLM-orchestrated path.

## Common commands

```bash
# End-to-end demo (both genre-profile and reference-match scenarios)
python3 run_demo.py

# Quick smoke test after any change to decision.py / dsp.py / analysis.py
python3 -c "
from mastering_agent import decision, orchestrator
r = orchestrator.run_mastering_loop('synth_mix.wav', 'output/smoke.wav',
    engine=decision.RuleBasedDecisionEngine(), max_iterations=4)
print(r.engine_used, r.warnings, r.final_features['crest_factor_db'])
"

# Regenerate synthetic test audio (only needed if you delete/modify it)
python3 make_synthetic_mix.py

# Reinstall after editing pyproject.toml
pip install -e .
```

There is no formal test suite yet (`pytest` is listed as a dev dependency
for when one is added) — validate changes via `run_demo.py` and the smoke
test above, checking the printed convergence log and `result.warnings`.

## Non-negotiable constraints — do not weaken these without being asked

1. **Crest-factor hard floor** (`decision.MIN_CREST_FACTOR_DB = 6.0`,
   enforced in `enforce_hard_constraints()`). This must run *after* every
   decision engine's proposal (LLM or rule-based), on *every* iteration, not
   just the final render. This is the project's core differentiator versus
   commercial "loudness-war" AI masters — never bypass it, raise the limit
   silently, or let a decision engine's output skip this function.
2. **True-peak ceiling** (`orchestrator.TRUE_PEAK_HARD_CEILING = -1.0`
   dBTP). Enforced in three places by design (limiter's own oversampled
   check, `enforce_hard_constraints`, and the orchestrator's final-render
   safety trim) — this redundancy is intentional belt-and-suspenders, not
   dead code to be "cleaned up".
3. **The Gemini engine must always degrade gracefully.** Any failure
   (missing key, network error, malformed JSON) must fall through to
   `RuleBasedDecisionEngine` — never let a mastering run hang or crash on an
   external API call. Preserve `engine_used` / `used_fallback_last_call`
   reporting so it's always transparent which engine actually ran.
4. **State management in the orchestrator loop**: each iteration must reason
   from the *previously rendered* state (`feats = post_feats` at the end of
   the loop body), not the dry mix repeatedly. If you touch
   `orchestrator.py`, keep this invariant — it was a real bug once (fixed:
   the loop originally re-analyzed the dry buffer every iteration, causing
   the loudness correction to overshoot instead of converging).

## Code style / conventions

- Dataclasses for all DSP/decision parameter contracts (`dsp.ChainParams`,
  `analysis.AudioFeatures`, etc.) — keep new parameters typed and defaulted,
  not raw dicts, so the decision engines and DSP chain stay decoupled.
- Every DSP stage in `dsp.py` is a pure function `(audio, params, sr) ->
  audio`; don't introduce hidden state or global config.
- Decision engines only ever produce `ChainParams` — they must never call
  into `dsp.py` directly or perform I/O themselves.
- Prefer adding new hard safety checks to `enforce_hard_constraints()` over
  trying to prompt-engineer an LLM into compliance.
- New genre profiles go in `analysis.GENRE_PROFILES`; keep the schema
  (`target_band_db`, `target_lufs`, `min_crest_db`, `target_width`)
  consistent so `profile_genre()` doesn't need special-casing.

## Known limitations (see README.md for full list)

- `GENRE_PROFILES` is hand-authored, not learned from a real corpus —
  `profile_genre()`'s nearest-centroid contract is designed so a trained
  k-means/GMM model can be swapped in without changing callers.
- Dynamic EQ and limiter envelope followers are explicit Python loops
  (correct but not optimized for real-time/very long files); a production
  port would move these to Numba or a compiled extension.
- No batch/album-processing orchestration yet — `run_mastering_loop`
  handles one file at a time.

If you (Claude) are extending this project, update this file, `README.md`,
and `INSTALL.md` together when you change setup steps, architecture, or
constraints — they should never drift out of sync.
