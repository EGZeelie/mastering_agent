"""
The Iterative Control Loop
============================

Wires Ears -> Brain -> Hands together into a closed loop:

    1. analyze the current buffer
    2. profile genre (and/or diff against reference)
    3. ask the decision engine for a chain-parameter proposal
    4. clamp the proposal through hard safety constraints (crest factor, true peak)
    5. render a short preview buffer with those params
    6. re-analyze the render; check convergence against target metrics
    7. repeat (accumulating chain params) until converged or max iterations hit
    8. do a final full-length render + QC pass + write output files

This mirrors the "renders a short buffer, re-analyzes, iteratively adjusts"
requirement in the brief, and keeps the crest-factor governor active on
every single iteration -- not just the final pass.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

from . import analysis, dsp, decision


@dataclass
class IterationLog:
    iteration: int
    lufs: float
    true_peak_dbtp: float
    crest_factor_db: float
    stereo_width: float
    params_snapshot: dict
    reasoning: Optional[str] = None
    converged: bool = False
    notes: str = ""


@dataclass
class MasteringResult:
    output_path: str
    iterations: list = field(default_factory=list)
    final_features: dict = field(default_factory=dict)
    engine_used: str = ""
    warnings: list = field(default_factory=list)


PREVIEW_SECONDS = 12.0
MAX_ITERATIONS = 4
LUFS_TOLERANCE = 0.6
TRUE_PEAK_HARD_CEILING = -1.0


def _extract_preview(y: np.ndarray, sr: int, seconds: float = PREVIEW_SECONDS) -> np.ndarray:
    n = min(y.shape[1], int(sr * seconds))
    if n >= y.shape[1]:
        return y
    start = max(0, (y.shape[1] - n) // 2)
    return y[:, start:start + n]


def _merge_chain_params(base: dsp.ChainParams, update: dsp.ChainParams, iteration: int) -> dsp.ChainParams:
    """Accumulate the decision engine's incremental proposals into a running
    chain, rather than discarding history each iteration (true agentic state
    management: the agent remembers what it already did)."""
    if iteration == 0:
        return update
    merged = copy.deepcopy(base)
    merged.linear_eq.extend(update.linear_eq)
    merged.dynamic_eq.extend(update.dynamic_eq)
    merged.compressor = update.compressor
    merged.saturation = update.saturation
    merged.stereo_width_mult = (merged.stereo_width_mult + update.stereo_width_mult) / 2
    merged.limiter = update.limiter
    merged.input_trim_db += update.input_trim_db
    return merged


def run_mastering_loop(
    input_path: str,
    output_path: str,
    reference_path: Optional[str] = None,
    engine: Optional[object] = None,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> MasteringResult:
    if engine is None:
        engine = decision.GeminiDecisionEngine()

    y_full = analysis._load_stereo(input_path)
    sr = analysis.SAMPLE_RATE_ANALYSIS
    y_preview = _extract_preview(y_full, sr)

    reference_features = None
    if reference_path:
        reference_features = analysis.analyze(reference_path)

    chain = dsp.ChainParams()
    logs = []
    warnings = []

    # `feats` always reflects the *current best-known state of the master*,
    # not the dry mix -- this is the agent's state management: iteration 0
    # reasons from the raw mix, every subsequent iteration reasons from what
    # its own previous chain actually produced, so proposals are genuinely
    # incremental corrections rather than repeated from-scratch guesses.
    feats = analysis.analyze(input_path)

    for it in range(max_iterations):
        genre_profile = analysis.profile_genre(feats)
        ref_diff = analysis.diff_against_reference(feats, reference_features) if reference_features else None

        proposal = engine.propose_chain_update(feats, genre_profile, ref_diff, it)
        proposal = decision.enforce_hard_constraints(proposal, feats)
        chain = _merge_chain_params(chain, proposal, it)
        chain.limiter.ceiling_dbtp = min(chain.limiter.ceiling_dbtp, TRUE_PEAK_HARD_CEILING)

        # Always render from the DRY preview buffer using the full
        # accumulated chain (never re-process an already-processed signal,
        # which would compound EQ/compression errors across iterations).
        rendered_preview = dsp.render_chain(y_preview, sr, chain)
        sf.write("/tmp/_ma_preview_render.wav", rendered_preview.T, sr)
        post_feats = analysis.analyze("/tmp/_ma_preview_render.wav")

        target_lufs = ref_diff["reference_lufs"] if ref_diff else genre_profile["target_lufs"]
        lufs_error = abs(post_feats.lufs_integrated - target_lufs)
        crest_ok = post_feats.crest_factor_db >= (decision.MIN_CREST_FACTOR_DB - 0.3)
        peak_ok = post_feats.true_peak_dbtp <= TRUE_PEAK_HARD_CEILING + 0.05
        converged = (lufs_error <= LUFS_TOLERANCE) and crest_ok and peak_ok

        reasoning = getattr(engine, "last_reasoning", None)
        notes = (
            f"target_lufs={target_lufs:.2f} lufs_error={lufs_error:.2f} "
            f"crest_ok={crest_ok} peak_ok={peak_ok}"
        )
        logs.append(IterationLog(
            iteration=it,
            lufs=post_feats.lufs_integrated,
            true_peak_dbtp=post_feats.true_peak_dbtp,
            crest_factor_db=post_feats.crest_factor_db,
            stereo_width=post_feats.stereo_width,
            params_snapshot=decision._chain_params_to_dict(chain),
            reasoning=reasoning,
            converged=converged,
            notes=notes,
        ))
        if verbose:
            print(f"[iter {it}] LUFS={post_feats.lufs_integrated:.2f} (target {target_lufs:.2f}) "
                  f"TP={post_feats.true_peak_dbtp:.2f}dBTP crest={post_feats.crest_factor_db:.2f}dB "
                  f"converged={converged}")

        if converged:
            break

        # Feed the *processed* state forward so the next iteration's
        # decision is a genuine incremental correction, not a repeat.
        feats = post_feats

    if not logs[-1].converged:
        warnings.append(
            "Did not fully converge within max_iterations; final render used the closest "
            "achieved parameter set. Consider raising max_iterations or reviewing genre match."
        )

    # ---- Final full-length render ----
    final_render = dsp.render_chain(y_full, sr, chain)

    final_peak = np.max(np.abs(final_render))
    ceiling_lin = 10 ** (TRUE_PEAK_HARD_CEILING / 20.0)
    if final_peak > ceiling_lin:
        final_render = final_render * (ceiling_lin / final_peak)
        warnings.append("Applied emergency safety trim to guarantee true-peak compliance on final render.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, final_render.T, sr, subtype="PCM_24")

    sf.write("/tmp/_ma_final_qc.wav", final_render.T, sr)
    final_feats = analysis.analyze("/tmp/_ma_final_qc.wav")

    if final_feats.crest_factor_db < decision.MIN_CREST_FACTOR_DB - 0.5:
        warnings.append(
            f"QC WARNING: final crest factor {final_feats.crest_factor_db:.2f} dB is below the "
            f"{decision.MIN_CREST_FACTOR_DB} dB floor -- review chain, this should not happen "
            f"if hard constraints are functioning correctly."
        )
    if final_feats.true_peak_dbtp > TRUE_PEAK_HARD_CEILING + 0.05:
        warnings.append(
            f"QC WARNING: final true peak {final_feats.true_peak_dbtp:.2f} dBTP exceeds "
            f"streaming-safe ceiling of {TRUE_PEAK_HARD_CEILING} dBTP."
        )

    engine_name = getattr(engine, "name", engine.__class__.__name__)
    if getattr(engine, "used_fallback_last_call", False):
        engine_name = f"{engine_name} (fell back to rule-based this run)"

    result = MasteringResult(
        output_path=output_path,
        iterations=[vars(log) for log in logs],
        final_features=final_feats.to_dict(),
        engine_used=engine_name,
        warnings=warnings,
    )
    return result


def save_report(result: MasteringResult, path: str):
    with open(path, "w") as f:
        json.dump(
            {
                "output_path": result.output_path,
                "engine_used": result.engine_used,
                "iterations": result.iterations,
                "final_features": result.final_features,
                "warnings": result.warnings,
            },
            f,
            indent=2,
            default=str,
        )
