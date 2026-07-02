# Install Instructions

Two supported ways to install, depending on how you want to use it. Both
have been verified from a clean virtual environment in this sandbox.

- **Option A** — install as a proper Python package (recommended if you'll
  `import mastering_agent` from other code, e.g. an MCP server).
- **Option B** — just install dependencies and run the scripts in place
  (fastest if you only want to run `run_demo.py` / master files directly).

---

## Requirements

- **Python 3.9+** (developed/tested on 3.13)
- **No system/OS packages required** — the DSP stack is pure Python +
  NumPy/SciPy, and audio I/O goes through `soundfile` (libsndfile is bundled
  by the `soundfile` wheel on all major platforms). `ffmpeg` is **not**
  required for WAV files; only install it separately if you need to feed the
  agent compressed formats like MP3 (see *Optional* below).
- Internet access only if you want the real Gemini decision engine — the
  agent works fully offline otherwise (deterministic rule-based fallback).

---

## Option A: Install as a package

```bash
cd mastering_agent

# 1. Create an isolated environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the package + core dependencies
pip install -e .

# 3. (Optional) Install the Gemini extra to enable real LLM orchestration
pip install -e ".[gemini]"

# 4. Verify the install
python3 -c "from mastering_agent import analysis, dsp, decision, orchestrator; print('OK')"
```

You can now `import mastering_agent` from anywhere (e.g. an MCP server, a
batch-processing script, a notebook) as long as the venv is active.

---

## Option B: Run in place (no package install)

```bash
cd mastering_agent

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Then run scripts directly from this directory:

```bash
python3 make_synthetic_mix.py   # generates test audio (synth_mix.wav, reference_master.wav)
python3 run_demo.py              # runs the full closed-loop demo, writes ./output/
```

`requirements.txt` includes `google-genai` by default; comment it out if you
only want the offline rule-based engine and prefer a lighter install.

---

## Enabling the Gemini decision engine (optional)

Without an API key, the agent automatically and transparently falls back to
the deterministic rule-based decision engine — nothing breaks, but you don't
get real LLM reasoning. To enable it:

```bash
pip install google-genai      # if not already installed via [gemini] extra
export GEMINI_API_KEY=your_key_here     # or: export GOOGLE_API_KEY=your_key_here
```

Then any run of `run_demo.py` or `orchestrator.run_mastering_loop(...)` with
`decision.GeminiDecisionEngine()` (the default) will use the real model.
Check `result.engine_used` in the output/report to confirm which engine
actually ran (it reports honestly if it had to fall back).

---

## Optional: MP3/compressed-format input support

The core pipeline reads/writes WAV (via `soundfile`, no extra install
needed). If you need to feed it MP3, AAC, or other compressed formats,
install `ffmpeg` so `librosa`'s loader can decode them:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

Output masters are always written as WAV (24-bit PCM) regardless of input
format, which is the correct source-of-truth format for a mastering
pipeline — do any final lossy encoding (MP3/AAC for distribution) as a
separate downstream step, not inside the mastering chain itself.

---

## Quick smoke test

After either install option, confirm everything works end to end:

```bash
python3 make_synthetic_mix.py
python3 -c "
from mastering_agent import decision, orchestrator
result = orchestrator.run_mastering_loop(
    'synth_mix.wav', 'output/smoke_test.wav',
    reference_path=None, engine=decision.RuleBasedDecisionEngine(), max_iterations=4,
)
print('engine used:', result.engine_used)
print('warnings:', result.warnings)
print('final LUFS/TP/crest:', result.final_features['lufs_integrated'],
      result.final_features['true_peak_dbtp'], result.final_features['crest_factor_db'])
"
```

Expected: it converges within a few iterations, reports no warnings, and
produces `output/smoke_test.wav` with true peak ≤ -1.0 dBTP and crest factor
comfortably above the 6 dB hard floor.

---

## Using with Claude Code

This repo includes a [`CLAUDE.md`](CLAUDE.md) file with project-specific
context (architecture summary, common commands, and — importantly — the
non-negotiable safety constraints like the crest-factor floor and true-peak
ceiling that Claude Code should never weaken). Claude Code reads it
automatically.

```bash
# 1. Install Claude Code (if you haven't already)
npm install -g @anthropic-ai/claude-code

# 2. Set up the project environment first (see Option A/B above)
cd mastering_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gemini]"        # or: pip install -r requirements.txt

# 3. Launch Claude Code from the project root
claude
```

Once running, Claude Code will pick up `CLAUDE.md` for context automatically.
A few tips specific to this project:

- **No API key needed to let Claude test changes.** The mastering pipeline
  runs fully offline via the rule-based fallback engine, so Claude can run
  `python3 run_demo.py` or the smoke test in `CLAUDE.md` to validate edits
  without needing `GEMINI_API_KEY` configured.
- **If you *do* want Claude testing the real Gemini decision path**, export
  the key in the same shell before launching `claude` so it's inherited by
  any commands Claude runs:
  ```bash
  export GEMINI_API_KEY=your_key_here
  claude
  ```
- **Ask Claude to check `result.warnings` and `result.engine_used`** after
  any change — these are the fastest way to confirm the hard constraints
  (crest factor, true peak) are still holding and that the decision engine
  behaved as expected.
- If you want Claude Code to run commands without prompting for approval
  each time (useful for iterating quickly on DSP tuning), use
  `claude --dangerously-skip-permissions` inside a disposable/sandboxed
  environment only — not recommended on a machine with sensitive data.

---

## Troubleshooting


| Symptom | Likely cause / fix |
|---|---|
| `ImportError: No module named 'mastering_agent'` | You used Option B (script mode) but tried to `import` from a different working directory. Either run scripts from inside `mastering_agent/`, or use Option A (`pip install -e .`) to make the package importable anywhere. |
| `soundfile.LibsndfileError` on load | Input file format isn't supported by libsndfile directly (e.g. MP3). Install `ffmpeg` (see above) so librosa can transcode on load. |
| Gemini engine silently falls back every time | Check `echo $GEMINI_API_KEY` is actually set in the shell you're running from, and that `google-genai` (or `google-generativeai`) is installed in the **active** venv. `result.engine_used` will tell you explicitly if/why it fell back. |
| Renders are slow on long files | Expected for this prototype — dynamic EQ/limiter envelope followers are plain Python loops. A 30s stereo file renders in ~5s; a full 4-minute track will take proportionally longer. See README's *Known limitations* for the production optimization path (Numba/C++ extension). |
