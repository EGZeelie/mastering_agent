"""Generate the GitHub social-preview image (1280x640) for the repo.

Reproducible: draws the real waveform envelope from synth_mix.wav over a
dark studio-themed gradient, with the Ears -> Brain -> Hands tagline and the
project's headline safety stat (crest-factor floor).

Usage:
    python assets/make_social_preview.py
Writes: assets/social_preview.png
"""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(ROOT, "synth_mix.wav")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_preview.png")

# Palette (studio dark + accent teal/amber)
BG_TOP = (13, 17, 23)       # GitHub dark
BG_BOT = (22, 27, 34)
ACCENT = (56, 211, 190)     # teal
ACCENT2 = (240, 185, 66)    # amber
WAVE = (56, 211, 190)
WAVE_DIM = (38, 92, 88)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def vertical_gradient(w: int, h: int, top, bot) -> Image.Image:
    base = Image.new("RGB", (w, h), top)
    top_a = np.array(top, dtype=np.float64)
    bot_a = np.array(bot, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, h)[:, None]
    col = (top_a[None, :] * (1 - ramp) + bot_a[None, :] * ramp).astype(np.uint8)
    grad = np.repeat(col[:, None, :], w, axis=1)
    return Image.fromarray(grad, "RGB")


def waveform_envelope(path: str, buckets: int) -> np.ndarray:
    audio, _ = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1)
    n = len(mono) // buckets
    trimmed = mono[: n * buckets].reshape(buckets, n)
    env = np.abs(trimmed).max(axis=1)
    env = env / (env.max() + 1e-9)
    # gentle gamma so quiet detail is visible
    return env ** 0.7


def main() -> None:
    img = vertical_gradient(W, H, BG_TOP, BG_BOT)
    d = ImageDraw.Draw(img)

    # --- Waveform band (center) ---
    buckets = 210
    env = waveform_envelope(AUDIO, buckets)
    band_cy = 340
    band_h = 150
    margin = 90
    usable = W - 2 * margin
    bw = usable / buckets
    for i, e in enumerate(env):
        x = margin + i * bw
        bar_h = max(2, e * band_h)
        # mirror around center line, teal->dim gradient by height
        mix = e
        col = tuple(int(WAVE_DIM[c] + (WAVE[c] - WAVE_DIM[c]) * mix) for c in range(3))
        d.rectangle([x, band_cy - bar_h, x + bw * 0.62, band_cy + bar_h], fill=col)

    # subtle center line
    d.line([margin, band_cy, W - margin, band_cy], fill=(30, 40, 46), width=1)

    # --- Title ---
    f_title = font("segoeuib.ttf", 76)
    f_sub = font("segoeui.ttf", 30)
    f_chip = font("segoeuib.ttf", 26)
    f_mono = font("consolab.ttf", 24)
    f_foot = font("segoeui.ttf", 22)

    d.text((margin, 70), "Autonomous Mastering Agent", font=f_title, fill=TEXT)
    d.text(
        (margin, 162),
        "Closed-loop AI mastering  —  analyze → decide → render → re-analyze",
        font=f_sub,
        fill=MUTED,
    )

    # --- Ears / Brain / Hands chips ---
    chips = [("\U0001F442  Ears", ACCENT), ("\U0001F9E0  Brain", ACCENT2), ("✋  Hands", ACCENT)]
    cx = margin
    cy = 230
    for label, acc in chips:
        tw = d.textlength(label, font=f_chip)
        pad = 18
        box = [cx, cy, cx + tw + 2 * pad, cy + 46]
        d.rounded_rectangle(box, radius=12, outline=acc, width=2)
        d.text((cx + pad, cy + 8), label, font=f_chip, fill=TEXT)
        cx = box[2] + 16

    # --- Footer stat strip ---
    strip_y = 500
    d.rounded_rectangle([margin, strip_y, W - margin, strip_y + 88], radius=16,
                        fill=(20, 26, 32), outline=(38, 47, 56), width=1)
    stats = [
        ("CREST FLOOR", "≥ 6.0 dB"),
        ("TRUE-PEAK CEIL", "≤ -1.0 dBTP"),
        ("ENGINE", "Gemini → rule-based"),
        ("FALLBACK", "always graceful"),
    ]
    seg = (W - 2 * margin) / len(stats)
    for i, (k, v) in enumerate(stats):
        sx = margin + i * seg + 24
        d.text((sx, strip_y + 16), k, font=f_foot, fill=MUTED)
        d.text((sx, strip_y + 44), v, font=f_mono, fill=ACCENT)
        if i:
            lx = margin + i * seg
            d.line([lx, strip_y + 18, lx, strip_y + 70], fill=(38, 47, 56), width=1)

    # top accent hairline
    d.line([0, 0, W, 0], fill=ACCENT, width=4)

    img.save(OUT, "PNG")
    print(f"wrote {OUT}  ({W}x{H})")


if __name__ == "__main__":
    main()
