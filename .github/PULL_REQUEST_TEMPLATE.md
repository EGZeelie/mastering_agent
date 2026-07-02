## Summary

<!-- What does this PR change, and why? -->

## Which layer(s) does this touch?

- [ ] Ears (`analysis.py`)
- [ ] Brain (`decision.py`)
- [ ] Hands (`dsp.py`)
- [ ] Loop (`orchestrator.py`)
- [ ] Tests / CI / tooling
- [ ] Docs only

## Safety checklist (required for anything touching `decision.py`, `dsp.py`, or `orchestrator.py`)

- [ ] I did **not** weaken or bypass `decision.MIN_CREST_FACTOR_DB` or `enforce_hard_constraints()`.
- [ ] I did **not** raise `TRUE_PEAK_CEILING_DBTP` / `orchestrator.TRUE_PEAK_HARD_CEILING` above -1.0 dBTP.
- [ ] If I added a new decision engine or DSP stage, it produces `ChainParams` / operates on `(audio, params, sr)` only — no hidden state, no direct I/O from decision code.
- [ ] `pytest tests/ -v` passes locally.
- [ ] `ruff check .` passes locally.
- [ ] `python run_demo.py` still converges with no unexpected warnings.

## How was this tested?

<!-- e.g. "Added test_x in test_decision.py"; "Ran run_demo.py with a real GEMINI_API_KEY and confirmed engine_used == 'gemini'"; "Manually verified against my_real_mix.wav" -->

## Related issues

<!-- Closes #123 -->
