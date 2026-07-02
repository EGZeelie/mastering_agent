"""
Autonomous Mastering Agent
==========================

A closed-loop, agentic mastering system:

  Ears  (analysis.py)     -> feature extraction, genre/reference profiling, problem detection
  Brain (decision.py)     -> LLM-orchestrated (Gemini) or rule-based parameter deltas
  Hands (dsp.py)          -> headless DSP chain: linear-phase EQ -> dynamic EQ ->
                              VCA bus compression -> harmonic saturation -> true-peak limiter
  Loop  (orchestrator.py) -> iterative render/re-analyze convergence with a hard
                              crest-factor constraint to prevent over-limiting

See README.md for architecture notes and run_demo.py for an end-to-end example.
"""

__version__ = "0.1.0"
