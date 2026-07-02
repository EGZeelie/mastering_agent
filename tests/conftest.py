"""Shared pytest fixtures for the mastering agent test suite.

Uses short (few-second) synthetic audio so the full suite runs in seconds,
not minutes -- the same generators used for the human-facing demo
(make_synthetic_mix.py) are reused here with a short duration and a fixed
seed for reproducibility.
"""
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from make_synthetic_mix import build_flawed_mix, build_reference_master, SR

TEST_DURATION = 3.0  # seconds -- short enough for a fast CI suite


@pytest.fixture(scope="session")
def sample_rate():
    return SR


@pytest.fixture()
def flawed_mix_path(tmp_path):
    np.random.seed(42)
    audio = build_flawed_mix(duration=TEST_DURATION, sr=SR)
    path = tmp_path / "flawed_mix.wav"
    sf.write(str(path), audio, SR, subtype="PCM_24")
    return str(path)


@pytest.fixture()
def reference_master_path(tmp_path):
    np.random.seed(43)
    audio = build_reference_master(duration=TEST_DURATION, sr=SR)
    path = tmp_path / "reference_master.wav"
    sf.write(str(path), audio, SR, subtype="PCM_24")
    return str(path)


@pytest.fixture()
def silence_path(tmp_path):
    """A degenerate edge case: near-silent stereo audio."""
    n = int(SR * TEST_DURATION)
    audio = (np.random.randn(n, 2) * 1e-6).astype(np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), audio, SR, subtype="PCM_24")
    return str(path)
