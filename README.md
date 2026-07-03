# Autonomous Mastering Agent — Prototype

[![CI](https://github.com/EGZeelie/mastering_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/EGZeelie/mastering_agent/actions/workflows/ci.yml)
[![CodeQL](https://github.com/EGZeelie/mastering_agent/actions/workflows/codeql.yml/badge.svg)](https://github.com/EGZeelie/mastering_agent/actions/workflows/codeql.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-lightgrey)](LICENSE)

A closed-loop, agentic mastering pipeline: **Ears → Brain → Hands**, run
iteratively until the master converges on target loudness/tone/dynamics —
with a hard, code-enforced crest-factor floor so the agent can never
"solve" loudness by ruthlessly squashing transients.

This is a working server-side Python prototype, built and validated end to
end in this sandbox against synthetic test audio (no real mixdown was
available in this session — see *Testing* below).

**➡️ See [INSTALL.md](INSTALL.md) for setup instructions** (package install,
script-mode install, enabling the real Gemini engine, troubleshooting, and
[Claude Code](CLAUDE.md) integration).

**➡️ See [GITHUB_SETUP.md](GITHUB_SETUP.md)** for the full GitHub repository
configuration checklist (description, topics, branch protection, CI/CodeQL/
Dependabot setup).

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              orchestrator.py                │
                 │   iterative render → analyze → decide loop  │
                 └───────────────┬───────────────────────────--┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                             │
     ▼                           ▼                             ▼
 analysis.py                decision.py                    dsp.py
 "The Ears"                 "The Brain"                    "The Hands"
 ─────────────              ─────────────                  ─────────────
 - LUFS (BS.1770)           - GeminiDecisionEngine          - Linear-phase EQ (FIR)
 - True peak (4x OS)          (LLM-orchestrated,            - Surgical dynamic EQ
 - Crest factor                structured JSON output)      - VCA-style bus compressor
 - Stereo width/corr        - RuleBasedDecisionEngine        (feed-forward RMS, soft knee)
 - Band energy (8 bands)      (deterministic fallback/       - Harmonic saturation (tanh)
 - Spectral centroid/tilt     offline baseline)              - Mid/side stereo width
 - Problem detection:       - enforce_hard_constraints()     - True-peak brickwall limiter
   sub-bass buildup,          <- CREST FACTOR FLOOR LIVES     (lookahead, oversampled)
   harsh resonances             HERE, runs after every
 - Genre profiling             engine's proposal
   (nearest-centroid match
   against a target-curve
   bank; swappable for a
   trained k-means/GMM model)
 - Reference-track diff
   (loudness/tone/width
   differential)
```

### The control loop (`orchestrator.run_mastering_loop`)

1. Analyze the current best-known state of the master (dry mix on iteration 0).
2. Profile genre and/or diff against a reference track.
3. Ask the decision engine (Gemini or rule-based) for a `ChainParams` proposal.
4. Run the proposal through `enforce_hard_constraints` — the safety governor.
5. Merge into the accumulated chain, render a ~12s preview buffer from the
   *dry* signal using the full accumulated chain (never reprocess
   already-processed audio — that compounds EQ/compression error).
6. Re-analyze the render; check convergence (LUFS within tolerance, crest
   factor above floor, true peak under ceiling).
7. Feed the *processed* state forward as input to the next iteration — this
   is genuine agentic state management: iteration 2 reasons about what
   iteration 1 actually produced, not a repeated from-scratch guess.
8. Repeat until converged or `max_iterations` reached, then do a full-length
   final render + QC pass (with an emergency true-peak safety trim that can
   never be bypassed) and write the output file + a JSON decision log.

## The hard crest-factor constraint

This was the single most important requirement in the design brief, and
it's implemented as **code that runs after any LLM output**, not as a prompt
instruction the model could ignore or drift from:

```python
# decision.py
MIN_CREST_FACTOR_DB = 6.0  # absolute floor, independent of genre/reference

def enforce_hard_constraints(params, features, target_crest_floor=MIN_CREST_FACTOR_DB):
    ...
    projected_crest = features.crest_factor_db - projected_crest_loss
    if projected_crest < target_crest_floor:
        # ease compressor ratio/mix proportionally to the deficit
        ...
```

I verified this directly: feeding a deliberately abusive proposal (10:1
ratio, 100% wet, -0.2 dBTP ceiling) through `enforce_hard_constraints`
produces `ratio=4.0, mix=0.9, ceiling=-1.0 dBTP` — the governor clamps it
every time, regardless of what any LLM (or bug) proposes upstream. This
also runs on **every iteration**, not just the final pass, and is separate
from the true-peak safety net that runs again at final-render time.

## LLM orchestration (Gemini)

`decision.GeminiDecisionEngine` sends the full analysis JSON (features +
genre profile + reference diff + hard-constraint reminders) to Gemini with
a system prompt that mandates a structured JSON schema for the chain
parameters (this is the "logical orchestration" layer from the brief — swap
in an MCP tool-calling loop by exposing `analyze`, `propose_chain_update`,
and `render_chain` as MCP tools; the contracts are already clean function
boundaries for that).

**In this sandbox there is no `GEMINI_API_KEY`/`GOOGLE_API_KEY` configured**,
so every demo run in this session used the automatic fallback:
`RuleBasedDecisionEngine`. This is by design, not a shortcut — the engine
reports honestly which brain actually ran (see `engine_used` in the JSON
report, e.g. `"gemini(no_api_key->rule_based_fallback) (fell back to
rule-based this run)"`), and the fallback triggers transparently on *any*
failure (missing key, network error, malformed JSON from the model) so the
control loop never stalls waiting on a flaky external call.

To use real Gemini reasoning:
```bash
export GEMINI_API_KEY=your_key_here
pip install google-genai   # or: pip install google-generativeai
python run_demo.py
```

## Testing

There's a real `pytest` suite in `tests/` (28 tests, ~20s runtime) covering
the analysis engine, DSP chain, decision engines, and the full orchestrator
loop -- including the most important test in the project,
`test_hard_constraints_never_let_projected_crest_fall_below_floor`, which
sweeps a range of aggressive compressor ratios/mixes and asserts the
crest-factor floor is never violated after `enforce_hard_constraints()`
runs. Run it with:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

This is what CI runs on every push/PR (see `.github/workflows/ci.yml`),
across Python 3.10/3.11/3.12, alongside `ruff` linting and an end-to-end
`run_demo.py` smoke test.

No real mixdown was supplied for this session, so I also generated
**synthetic test audio** with known, deliberately-introduced problems
(`make_synthetic_mix.py`) for manual/exploratory validation:

- `synth_mix.wav` — an "unmastered" mix with a real sub-bass buildup (two
  sine tones at 45/55 Hz), a harsh resonance around 3.2 kHz, quiet level
  (~-17.6 LUFS), and a near-mono stereo image.
- `reference_master.wav` — a louder, wider, tonally-flatter track to
  exercise the reference-matching path.

Validated end to end:
- ✅ Problem detector correctly flags `sub_bass_buildup` and `harsh_resonance`.
- ✅ Genre profiler nearest-matches the synthetic EDM-like mix to
  `electronic_dance`.
- ✅ Full closed loop converges in 3 iterations in both genre-profile mode
  and reference-matching mode (see `output/report_*.json` for the full
  per-iteration decision log, including the fallback engine's reasoning
  string at each step).
- ✅ Final masters: correct duration, no NaNs/clipping, true peak ≤ -1.0
  dBTP, crest factor comfortably above the 6 dB floor (~9-11 dB in both
  scenarios).
- ✅ Hard-constraint governor verified directly against an intentionally
  abusive parameter proposal (see above).

Run it yourself:
```bash
cd mastering_agent
python make_synthetic_mix.py   # regenerate test audio (optional, already generated)
python run_demo.py             # runs both scenarios, writes output/*.wav + *.json
```

To master your own file:
```python
from mastering_agent import decision, orchestrator

result = orchestrator.run_mastering_loop(
    input_path="your_mix.wav",
    output_path="output/your_master.wav",
    reference_path="optional_reference.wav",  # or None for genre-profile mode
    engine=decision.GeminiDecisionEngine(),   # or decision.RuleBasedDecisionEngine()
    max_iterations=6,
)
orchestrator.save_report(result, "output/report.json")
print(result.warnings)
```

## Known limitations / honest caveats (this is a prototype, not production)

- **Genre bank is hand-authored, not learned.** `GENRE_PROFILES` in
  `analysis.py` is a stand-in for the EMA-style unsupervised clustering
  described in the brief. The calling contract (`profile_genre` does
  nearest-centroid matching) is identical to what a trained k-means/GMM
  model would need, so swapping in real learned centroids from a corpus of
  commercial masters is a drop-in change — I did not have a training corpus
  available in this session.
- **Dynamic EQ and limiter are implemented with explicit Python envelope
  loops** (readable, correct, and fast enough for this prototype — a 30s
  stereo file renders in ~5s) but a production system would move the
  sample-by-sample envelope followers to Numba/Cython or a C++ extension
  for real-time/large-batch use.
- **Linear-phase EQ uses a frequency-sampling FIR design** (2049 taps),
  which is standard for mastering-grade zero-phase-distortion EQ, but
  introduces ~21ms of latency (numtaps/2) — expected and fine for offline
  batch mastering, not suitable as-is for live monitoring.
- **Stereo width control cannot fix already-mono content** — it's a
  mid/side scale on *existing* side-channel energy, not a fake-stereo
  synthesizer. In testing, `synth_mix.wav` is almost perfectly mono
  (L/R correlation 0.99999) because that's how I built the test signal, so
  even a width multiplier of 1.4x barely moves the needle. This is correct,
  conservative DSP behavior (a mastering engineer shouldn't invent stereo
  info that was never in the mix) — I'm flagging it so it doesn't look like
  a bug when you inspect the reports.
- **True-peak limiter has two safety nets** (post-render sample-peak trim +
  a second oversampled re-check) because the resample_poly round-trip can
  introduce small reconstruction ripple; this is belt-and-suspenders
  engineering, not evidence the primary limiter design is unreliable.
- **No batch/album-processing orchestration yet** — the brief mentions this
  as a strength of the server-side approach; the current code masters one
  file at a time. Wiring `run_mastering_loop` into an async batch runner
  (with consistent reference-loudness targets across an album) is a natural
  next step, not yet built.

## File map

```
mastering_agent/
  __init__.py
  analysis.py        # The Ears: feature extraction, genre profiling, reference diff
  dsp.py              # The Hands: EQ / dynamic EQ / compressor / saturation / limiter
  decision.py         # The Brain: Gemini + rule-based engines, hard constraints
  orchestrator.py      # The Loop: iterative render/analyze/decide/converge
make_synthetic_mix.py  # generates synth_mix.wav + reference_master.wav for testing
run_demo.py             # end-to-end demo, both genre-profile and reference-match modes
output/                  # rendered masters + JSON decision logs from this session
INSTALL.md               # setup instructions (package install / script-mode / Gemini)
requirements.txt          # pinned-minimum dependency list for script-mode install
pyproject.toml             # package metadata for `pip install -e .`
```
