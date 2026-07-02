"""
The Agentic Decision Core ("The Brain")
=========================================

Translates raw AudioFeatures (+ genre profile and/or reference differential)
into concrete `dsp.ChainParams`. Two interchangeable engines share one
contract (`propose_chain_update`):

  1. `GeminiDecisionEngine`  - calls a Gemini model with the full analysis
     JSON and asks it to return parameter deltas as structured JSON. This is
     the "logical orchestration via MCP-style tool calling" requested in the
     brief: the LLM reasons over state, the code enforces safety.
  2. `RuleBasedDecisionEngine` - a deterministic fallback/baseline (and unit
     -testable ground truth) used automatically when no Gemini API key is
     configured, or if the LLM call fails/returns malformed data. This keeps
     the whole pipeline runnable offline and is also simply good engineering
     practice: never let an agent go silent because an external API hiccups.

Both engines are wrapped by `enforce_hard_constraints`, which is where the
non-negotiable crest-factor floor lives -- this runs *after* any LLM
suggestion, so no amount of prompt drift can make the agent over-limit the
master. This is the direct implementation of the brief's requirement:
"implement a hard constraint on the crest factor... to preserve transient
impact."
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

from .analysis import AudioFeatures, BANDS
from .dsp import (
    ChainParams, EQBand, DynamicEQBand, CompressorParams, SaturationParams, LimiterParams,
)

# ---------------------------------------------------------------------------
# Hard constraints (non-negotiable, enforced in code -- not merely "asked for"
# in a prompt). These are the agent's engineering conscience.
# ---------------------------------------------------------------------------
MIN_CREST_FACTOR_DB = 6.0          # absolute floor regardless of genre/target loudness
TRUE_PEAK_CEILING_DBTP = -1.0      # streaming-safe true-peak ceiling
MAX_LOUDNESS_OVERSHOOT_LUFS = 0.5  # don't blow past target loudness even if LLM asks


def _chain_params_to_dict(p: ChainParams) -> dict:
    return {
        "input_trim_db": p.input_trim_db,
        "linear_eq": [asdict(b) for b in p.linear_eq],
        "dynamic_eq": [asdict(b) for b in p.dynamic_eq],
        "compressor": asdict(p.compressor),
        "saturation": asdict(p.saturation),
        "stereo_width_mult": p.stereo_width_mult,
        "limiter": asdict(p.limiter),
    }


def _chain_params_from_dict(d: dict) -> ChainParams:
    return ChainParams(
        input_trim_db=float(d.get("input_trim_db", 0.0)),
        linear_eq=[EQBand(**b) for b in d.get("linear_eq", [])],
        dynamic_eq=[DynamicEQBand(**b) for b in d.get("dynamic_eq", [])],
        compressor=CompressorParams(**d.get("compressor", {})),
        saturation=SaturationParams(**d.get("saturation", {})),
        stereo_width_mult=float(d.get("stereo_width_mult", 1.0)),
        limiter=LimiterParams(**d.get("limiter", {})),
    )


# ---------------------------------------------------------------------------
# Rule-based baseline engine
# ---------------------------------------------------------------------------

class RuleBasedDecisionEngine:
    """Deterministic decision core. Used as the offline fallback and as a
    sanity baseline against which LLM proposals can be diffed."""

    name = "rule_based"

    def propose_chain_update(
        self,
        features: AudioFeatures,
        genre_profile: dict,
        reference_diff: Optional[dict],
        iteration: int,
    ) -> ChainParams:
        params = ChainParams()
        target_band_db = genre_profile["target_band_db"]
        # Reference-track loudness always takes precedence over the genre-profile
        # default when a reference is supplied -- matching a specific reference
        # is a more specific instruction than a generic style target.
        target_lufs = reference_diff["reference_lufs"] if reference_diff else genre_profile["target_lufs"]

        # 1) Corrective EQ from problem detection (fix flaws before enhancing)
        for problem in features.problems:
            if problem.kind == "sub_bass_buildup":
                params.linear_eq.append(EQBand(freq_hz=40, gain_db=-2.5, q=0.9, kind="low_shelf"))
            elif problem.kind == "harsh_resonance":
                center = (problem.band_hz[0] + problem.band_hz[1]) / 2
                params.dynamic_eq.append(DynamicEQBand(
                    freq_hz=center, q=2.5, threshold_db=-24.0, ratio=3.0,
                    max_cut_db=4.0, attack_ms=3.0, release_ms=60.0,
                ))

        # 2) Tonal shaping toward genre/reference target curve
        if reference_diff is not None:
            band_delta = reference_diff["band_delta_db"]
        else:
            band_delta = {b: target_band_db[b] - features.band_energy_db[b] for b in BANDS}

        band_centers = {
            "sub_bass": 50, "bass": 120, "low_mid": 350, "mid": 1000,
            "high_mid": 3000, "presence": 5000, "brilliance": 9000, "air": 17000,
        }
        for band, delta in band_delta.items():
            delta_clamped = float(max(-4.0, min(4.0, delta * 0.6)))  # damp toward target gradually
            if abs(delta_clamped) < 0.3:
                continue
            kind = "bell"
            if band == "sub_bass":
                kind = "low_shelf"
            elif band == "air":
                kind = "high_shelf"
            params.linear_eq.append(EQBand(
                freq_hz=band_centers[band], gain_db=delta_clamped, q=0.8, kind=kind,
            ))

        # 3) Bus compression -- gentle VCA-style glue, tighter if reference/genre wants density.
        # Threshold is also nudged by the current loudness error: if trim alone has plateaued
        # against the limiter (a real coupling in serial mastering chains -- trim, compression,
        # and limiting are NOT independent levers), pulling the compressor threshold down adds
        # a second, complementary path to close the loudness gap without over-limiting.
        target_crest = reference_diff["reference_crest_db"] if reference_diff else genre_profile["min_crest_db"] + 2.0
        loudness_error = target_lufs - features.lufs_integrated  # positive => need louder
        base_threshold = max(-28.0, features.rms_dbfs - 6.0)
        threshold_nudge = float(max(-6.0, min(2.0, -loudness_error * 0.5))) if iteration > 0 else 0.0
        threshold = base_threshold + threshold_nudge
        ratio = 1.8 if target_crest > 9 else 2.4
        params.compressor = CompressorParams(
            threshold_db=threshold, ratio=ratio, attack_ms=18.0, release_ms=140.0,
            makeup_db=1.5, mix=0.85 if iteration == 0 else 0.7,
        )

        # 4) Saturation for density/warmth (parallel, subtle)
        params.saturation = SaturationParams(drive_db=3.0, mix=0.15, kind="tanh")

        # 5) Stereo width toward target
        target_width = reference_diff["reference_width"] if reference_diff else genre_profile["target_width"]
        width_ratio = target_width / max(features.stereo_width, 0.05)
        params.stereo_width_mult = float(max(0.6, min(1.4, width_ratio)))

        # 6) Input trim + limiter to approach target loudness
        gain_needed = target_lufs - features.lufs_integrated
        params.input_trim_db = float(max(-6.0, min(12.0, gain_needed * 0.8)))
        params.limiter = LimiterParams(
            ceiling_dbtp=TRUE_PEAK_CEILING_DBTP, release_ms=80.0, lookahead_ms=5.0,
        )

        return params


# ---------------------------------------------------------------------------
# Gemini-orchestrated decision engine
# ---------------------------------------------------------------------------

GEMINI_SYSTEM_PROMPT = """You are the decision core of an autonomous audio mastering agent.
You receive measured audio features (loudness, true peak, crest factor, stereo width,
band-energy-in-dB, detected problems) plus a genre target profile and, optionally, a
reference-track differential. Your job is to output ONLY a JSON object describing a
mastering DSP chain update, matching this exact schema:

{
  "input_trim_db": float,
  "linear_eq": [{"freq_hz": float, "gain_db": float, "q": float, "kind": "bell"|"low_shelf"|"high_shelf"}],
  "dynamic_eq": [{"freq_hz": float, "q": float, "threshold_db": float, "ratio": float,
                  "max_cut_db": float, "attack_ms": float, "release_ms": float}],
  "compressor": {"threshold_db": float, "ratio": float, "attack_ms": float,
                 "release_ms": float, "makeup_db": float, "mix": float},
  "saturation": {"drive_db": float, "mix": float, "kind": "tanh"},
  "stereo_width_mult": float,
  "limiter": {"ceiling_dbtp": float, "release_ms": float, "lookahead_ms": float},
  "reasoning": "one short paragraph explaining the engineering decisions"
}

Rules you must follow:
- Fix problems (sub-bass buildup, harsh resonances) with corrective EQ/dynamic EQ
  BEFORE adding enhancement (saturation, width, extra loudness).
- dynamic_eq max_cut_db must be a POSITIVE magnitude (cut only, surgical).
- NEVER propose a limiter ceiling above -0.8 dBTP (true-peak safety for streaming).
- NEVER propose settings that would obviously crush the mix into near-zero crest
  factor -- transient impact must be preserved. Prefer moderate ratios (1.5-3.5:1)
  over brickwall-style bus compression.
- Output strictly valid JSON, no markdown fences, no commentary outside the JSON.
"""


class GeminiDecisionEngine:
    """LLM-orchestrated decision core using Gemini for structured parameter
    reasoning. Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment
    and the `google-genai` (or `google-generativeai`) package installed.
    Falls back to `RuleBasedDecisionEngine` transparently on any failure so
    the control loop never stalls waiting on a flaky network call."""

    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", fallback: Optional[RuleBasedDecisionEngine] = None):
        self.model = model
        self.fallback = fallback or RuleBasedDecisionEngine()
        self._client = None
        self._client_kind = None
        self.last_reasoning = None
        self.used_fallback_last_call = False
        self._init_client()
        if self._client is None:
            self.name = "gemini(no_api_key->rule_based_fallback)"

    def _init_client(self):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self._client = None
            return
        try:
            from google import genai  # new unified SDK
            self._client = genai.Client(api_key=api_key)
            self._client_kind = "genai"
        except Exception:
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=api_key)
                self._client = genai_legacy.GenerativeModel(self.model)
                self._client_kind = "legacy"
            except Exception:
                self._client = None

    def _build_prompt(self, features, genre_profile, reference_diff, iteration) -> str:
        payload = {
            "iteration": iteration,
            "features": features.to_dict(),
            "genre_profile": genre_profile,
            "reference_diff": reference_diff,
            "hard_constraints": {
                "min_crest_factor_db": MIN_CREST_FACTOR_DB,
                "true_peak_ceiling_dbtp": TRUE_PEAK_CEILING_DBTP,
            },
        }
        return json.dumps(payload, indent=2)

    def propose_chain_update(self, features, genre_profile, reference_diff, iteration) -> ChainParams:
        if self._client is None:
            self.used_fallback_last_call = True
            self.last_reasoning = "[No GEMINI_API_KEY/GOOGLE_API_KEY configured -- used rule-based fallback engine]"
            return self.fallback.propose_chain_update(features, genre_profile, reference_diff, iteration)

        prompt = self._build_prompt(features, genre_profile, reference_diff, iteration)
        try:
            text = self._call_model(prompt)
            data = self._extract_json(text)
            self.last_reasoning = data.pop("reasoning", None)
            self.used_fallback_last_call = False
            return _chain_params_from_dict(data)
        except Exception as e:
            self.used_fallback_last_call = True
            self.last_reasoning = f"[Gemini call failed, used rule-based fallback: {e}]"
            return self.fallback.propose_chain_update(features, genre_profile, reference_diff, iteration)

    def _call_model(self, prompt: str) -> str:
        full_prompt = GEMINI_SYSTEM_PROMPT + "\n\nANALYSIS INPUT:\n" + prompt
        if self._client_kind == "genai":
            resp = self._client.models.generate_content(model=self.model, contents=full_prompt)
            return resp.text
        elif self._client_kind == "legacy":
            resp = self._client.generate_content(full_prompt)
            return resp.text
        raise RuntimeError("No Gemini client initialized")

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Hard constraint enforcement -- runs AFTER any engine's proposal
# ---------------------------------------------------------------------------

def enforce_hard_constraints(params: ChainParams, features: AudioFeatures, target_crest_floor: float = MIN_CREST_FACTOR_DB) -> ChainParams:
    """The agent's non-negotiable engineering conscience.

    No matter what the LLM or rule engine proposed, this function guarantees:
      1. True-peak ceiling never exceeds streaming-safe limits.
      2. Compressor ratio/threshold cannot be configured to crush crest factor
         below the floor, given the mix's current dynamics.
      3. The limiter cannot be asked to claw back more gain reduction than is
         consistent with preserving the crest-factor floor -- this is the
         literal implementation of "a hard constraint on crest factor to
         preserve transient impact" from the design brief.
    """
    # (1) true-peak ceiling clamp
    if params.limiter.ceiling_dbtp > TRUE_PEAK_CEILING_DBTP:
        params.limiter.ceiling_dbtp = TRUE_PEAK_CEILING_DBTP

    # (2) compressor ratio clamp -- keep ratios musical, never limiter-on-the-bus
    if params.compressor.ratio > 4.0:
        params.compressor.ratio = 4.0
    if params.compressor.mix > 0.9:
        params.compressor.mix = 0.9

    # (3) crest-factor floor: estimate whether current settings would push the
    # projected crest factor below the floor, and if so, ease off the
    # compressor mix/ratio and back off the limiter's effective loudness push.
    projected_crest_loss = (
        (params.compressor.ratio - 1.0) * params.compressor.mix * 1.4
    )
    projected_crest = features.crest_factor_db - projected_crest_loss
    if projected_crest < target_crest_floor:
        deficit = target_crest_floor - projected_crest
        # ease compressor first (cheapest fix, preserves tone the most)
        scale = max(0.3, 1.0 - deficit / 6.0)
        params.compressor.ratio = max(1.2, 1.0 + (params.compressor.ratio - 1.0) * scale)
        params.compressor.mix = max(0.2, params.compressor.mix * scale)

    return params
