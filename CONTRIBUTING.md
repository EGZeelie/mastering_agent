# Contributing

This is currently a proprietary project (see [LICENSE](LICENSE)) — external
contributions are reviewed at the maintainer's discretion, but issues,
bug reports, and pull requests are still welcome from authorized
collaborators.

## Development setup

```bash
git clone https://github.com/EGZeelie/mastering_agent.git
cd mastering_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gemini]"
python3 make_synthetic_mix.py
```

See [INSTALL.md](INSTALL.md) for full setup details, including Claude Code
integration, and [README.md](README.md) for the architecture overview.

## Before opening a PR

```bash
pytest tests/ -v          # full test suite (should take ~20s)
ruff check .                # lint
python run_demo.py           # end-to-end smoke test, both scenarios
```

All three must pass. CI runs the same checks on Python 3.10/3.11/3.12 plus
a CodeQL security scan — PRs won't merge if any of these fail.

## The one rule that matters most

**Never weaken the hard safety constraints** in `decision.py`
(`MIN_CREST_FACTOR_DB`, `enforce_hard_constraints()`) or `orchestrator.py`
(`TRUE_PEAK_HARD_CEILING`). These exist specifically so the agent cannot
"solve" loudness by over-limiting and killing transient impact — see
README's *"The hard crest-factor constraint"* section for the full
rationale. If you're proposing a change that touches these, explain why in
the PR description and add a test in `tests/test_decision.py` proving the
floor still holds (see `test_hard_constraints_never_let_projected_crest_fall_below_floor`
for the pattern).

## Adding tests

- New DSP stages → `tests/test_dsp.py`, testing pure `(audio, params, sr) ->
  audio` behavior in isolation.
- New decision-engine logic or constraint changes → `tests/test_decision.py`.
- New analysis features (detectors, profiling) → `tests/test_analysis.py`.
- Anything affecting the full loop's convergence behavior →
  `tests/test_orchestrator.py`.

Use the short-duration fixtures in `tests/conftest.py`
(`flawed_mix_path`, `reference_master_path`, `silence_path`) rather than
generating new long audio files — the whole suite should stay well under a
minute.

## Commit / PR conventions

- Keep PRs scoped to one architectural layer where possible (see the PR
  template's checklist).
- Reference the specific `README.md` / `CLAUDE.md` sections you're updating
  if your change affects architecture, setup, or constraints — these three
  docs should never drift out of sync (see the note at the bottom of
  `CLAUDE.md`).
