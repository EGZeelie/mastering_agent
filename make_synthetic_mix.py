"""
Generates synthetic test audio for the mastering agent demo, since no real
mixdown was supplied. Produces:

  synth_mix.wav        - a deliberately flawed "unmastered" stereo mix:
                          muddy sub-bass, harsh ~3.2kHz resonance, quiet
                          (-18 LUFS), wide crest factor, narrow stereo image.
  reference_master.wav - a commercially-mastered-sounding reference track
                          (louder, wider, tonally balanced) to test the
                          reference-matching path.

Not intended to sound musical -- it's a controlled synthetic signal with
known, deliberately-introduced problems so the analysis engine's problem
detector and the decision core's corrective behavior can be verified
objectively.
"""
import numpy as np
import soundfile as sf

SR = 48000
DUR = 30.0


def make_drum_hits(n_samples, sr, bpm=120, kind="kick"):
    beat_len = int(sr * 60 / bpm)
    sig = np.zeros(n_samples)
    t_hit = np.linspace(0, 0.15, int(sr * 0.15))
    if kind == "kick":
        env = np.exp(-t_hit * 25)
        tone = np.sin(2 * np.pi * 60 * t_hit) * env
    else:  # snare/noise transient
        env = np.exp(-t_hit * 40)
        tone = (np.random.randn(len(t_hit)) * 0.5) * env
    for start in range(0, n_samples - len(tone), beat_len):
        sig[start:start + len(tone)] += tone
    return sig


def make_synth_pad(n_samples, sr, freqs):
    t = np.arange(n_samples) / sr
    sig = np.zeros(n_samples)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t) * 0.15
    return sig


def make_harsh_vocal_like(n_samples, sr):
    t = np.arange(n_samples) / sr
    carrier = np.sin(2 * np.pi * 220 * t)
    resonance = np.sin(2 * np.pi * 3200 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    formant_noise = np.random.randn(n_samples) * 0.02
    return carrier * 0.2 + resonance * 0.35 + formant_noise


def build_flawed_mix(duration=DUR, sr=SR):
    n = int(sr * duration)
    kick = make_drum_hits(n, sr, bpm=120, kind="kick")
    snare = make_drum_hits(n, sr, bpm=120, kind="snare")
    snare = np.roll(snare, int(sr * 60 / 120 / 2))
    pad = make_synth_pad(n, sr, [220, 330, 440])
    vocal = make_harsh_vocal_like(n, sr)

    sub_mud = np.sin(2 * np.pi * 45 * np.arange(n) / sr) * 0.35
    sub_mud += np.sin(2 * np.pi * 55 * np.arange(n) / sr) * 0.25

    broadband_bed = np.random.randn(n) * 0.06  # fills out mid/presence/brilliance/air like real instrument texture

    mono_mix = kick * 1.0 + snare * 0.8 + pad * 0.6 + vocal * 0.9 + sub_mud + broadband_bed

    left = mono_mix * 0.97 + np.random.randn(n) * 0.001
    right = mono_mix * 0.97 + np.random.randn(n) * 0.001
    stereo = np.stack([left, right])

    peak = np.max(np.abs(stereo))
    stereo = stereo / peak * 0.5  # leaves lots of headroom -> quiet, unmastered (~-18 LUFS-ish)

    return stereo.T.astype(np.float32)


def build_reference_master(duration=DUR, sr=SR):
    n = int(sr * duration)
    kick = make_drum_hits(n, sr, bpm=120, kind="kick")
    snare = make_drum_hits(n, sr, bpm=120, kind="snare")
    snare = np.roll(snare, int(sr * 60 / 120 / 2))
    pad = make_synth_pad(n, sr, [220, 330, 440, 660])
    vocal_carrier = np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 0.25
    formant_noise = np.random.randn(n) * 0.015

    broadband_bed = np.random.randn(n) * 0.05
    mono_mix = kick * 0.9 + snare * 0.7 + pad * 0.5 + vocal_carrier + formant_noise + broadband_bed

    t = np.arange(n) / sr
    side_component = np.sin(2 * np.pi * 330 * t) * 0.08 + np.random.randn(n) * 0.01
    left = mono_mix + side_component
    right = mono_mix - side_component

    stereo = np.stack([left, right])
    peak = np.max(np.abs(stereo)) + 1e-9
    stereo = stereo / peak * 0.9  # much hotter, mastered-sounding level

    # gentle soft clip / saturation to emulate a limited master with reasonable crest factor
    stereo = np.tanh(stereo * 1.3) / np.tanh(1.3)
    stereo = stereo * 0.9

    return stereo.T.astype(np.float32)



if __name__ == "__main__":
    np.random.seed(42)
    mix = build_flawed_mix()
    sf.write("synth_mix.wav", mix, SR, subtype="PCM_24")
    print("Wrote synth_mix.wav", mix.shape)

    ref = build_reference_master()
    sf.write("reference_master.wav", ref, SR, subtype="PCM_24")
    print("Wrote reference_master.wav", ref.shape)
