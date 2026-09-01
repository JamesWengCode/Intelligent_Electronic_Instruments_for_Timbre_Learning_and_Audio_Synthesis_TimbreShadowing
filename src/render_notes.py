import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyworld as pw

# Anti-pop / anti-click settings
FADE_IN_SECONDS = 0.004
FADE_OUT_SECONDS = 0.008

# Max distance (semitones) a note can be from the source pitch before
# falling back from vocoder pitch-shifting to DDSP -- A/B-tested by ear on
# real violin audio to this point with no degradation yet.
VOCODER_MAX_SEMITONES = 22.0


def midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def remove_dc(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio.astype(np.float32)
    return (audio - np.mean(audio)).astype(np.float32)


def fade_edges(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = audio.copy().astype(np.float32)

    fade_in_len = min(len(audio) // 2, max(1, int(sr * FADE_IN_SECONDS)))
    fade_out_len = min(len(audio) // 2, max(1, int(sr * FADE_OUT_SECONDS)))

    if fade_in_len > 1:
        fade_in = np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
        audio[:fade_in_len] *= fade_in

    if fade_out_len > 1:
        fade_out = np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)
        audio[-fade_out_len:] *= fade_out

    return audio.astype(np.float32)


def upsample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, len(x))
    target_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(target_x, source_x, x).astype(np.float32)


def upsample_2d_time(x: np.ndarray, target_len: int) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, x.shape[0])
    target_x = np.linspace(0.0, 1.0, target_len)

    out = np.zeros((target_len, x.shape[1]), dtype=np.float32)

    for h in range(x.shape[1]):
        out[:, h] = np.interp(target_x, source_x, x[:, h])

    out = out / (np.sum(out, axis=1, keepdims=True) + 1e-8)
    return out


def frequency_mapped_harmonic_frames(
    original_frames: np.ndarray,
    original_f0: float,
    new_f0: float,
    sr: int,
    brightness_compensation: float = 0.35,
) -> np.ndarray:
    """
    Convert learned harmonic weights from the original pitch to a new pitch.

    Old/simple method:
        harmonic 1 -> harmonic 1
        harmonic 2 -> harmonic 2
        ...

    New method:
        Treat the learned weights as a spectral envelope over frequency.
        Then sample that envelope at the new harmonic frequencies.
    """
    n_frames, old_n_harmonics = original_frames.shape

    nyquist = sr / 2.0
    max_new_harmonic = int((nyquist * 0.95) // new_f0)
    max_new_harmonic = max(1, max_new_harmonic)

    old_freqs = np.arange(1, old_n_harmonics + 1, dtype=np.float32) * original_f0
    new_freqs = np.arange(1, max_new_harmonic + 1, dtype=np.float32) * new_f0

    mapped = np.zeros((n_frames, max_new_harmonic), dtype=np.float32)

    for t in range(n_frames):
        old_weights = original_frames[t].astype(np.float32)

        old_log = np.log(old_weights + 1e-8)

        new_log = np.interp(
            new_freqs,
            old_freqs,
            old_log,
            left=old_log[0],
            right=old_log[-1],
        )

        new_weights = np.exp(new_log)

        ratio = new_f0 / original_f0
        # Anchor the damping curve to the original harmonic count, not the
        # smaller count left after transposing up -- otherwise a high note
        # with few surviving harmonics gets double-penalized (fewer
        # harmonics AND full damping), sounding unnaturally thin.
        new_harmonic_numbers = np.arange(1, max_new_harmonic + 1, dtype=np.float32)
        harmonic_index = (new_harmonic_numbers - 1.0) / max(1.0, old_n_harmonics - 1.0)
        harmonic_index = np.clip(harmonic_index, 0.0, 1.0)

        if ratio > 1.0:
            new_weights *= np.exp(
                -brightness_compensation * np.log(ratio) * harmonic_index
            )
        elif ratio < 1.0:
            new_weights *= np.exp(
                brightness_compensation * np.log(1.0 / ratio) * harmonic_index
            )

        # Harmonics beyond the trained count are pure extrapolation (flat,
        # unmeasured) -- roll them off so the brightness boost above
        # doesn't amplify that flat tail into buzz.
        overhang = new_harmonic_numbers - old_n_harmonics
        if np.any(overhang > 0):
            rolloff = np.where(overhang > 0, np.exp(-0.5 * np.maximum(overhang, 0)), 1.0)
            new_weights = new_weights * rolloff

        new_weights = new_weights + 1e-8
        new_weights = new_weights / np.sum(new_weights)

        mapped[t] = new_weights

    return mapped.astype(np.float32)


def amp_to_impulse_response(amp: np.ndarray, target_size: int) -> np.ndarray:
    """
    Frequency-sampling method: turn a per-frame nonnegative magnitude
    spectrum into a windowed, causal-centered real FIR impulse response.
    Mirrors train_timbre.py's torch implementation exactly (same algorithm,
    same Hann window convention) so rendering reproduces the trained filter.

    amp: [n_frames, n_mags]
    return: [n_frames, target_size]
    """
    impulse = np.fft.irfft(amp.astype(np.complex64), axis=-1)
    filter_size = impulse.shape[-1]

    impulse = np.roll(impulse, filter_size // 2, axis=-1)
    window = np.hanning(filter_size).astype(np.float32)
    impulse = impulse * window[None, :]

    if target_size > filter_size:
        impulse = np.pad(impulse, ((0, 0), (0, target_size - filter_size)))

    impulse = np.roll(impulse, -filter_size // 2, axis=-1)
    return impulse.astype(np.float32)


def fft_convolve(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    signal, kernel: [n_frames, n] (same trailing length n)
    Zero-padded FFT convolution; returns the causal linear-convolution
    result truncated back to length n.
    """
    n = signal.shape[-1]
    signal = np.pad(signal, ((0, 0), (0, n)))
    kernel = np.pad(kernel, ((0, 0), (n, 0)))

    output = np.fft.irfft(
        np.fft.rfft(signal, axis=-1) * np.fft.rfft(kernel, axis=-1),
        n=signal.shape[-1],
        axis=-1,
    )
    return output[:, n:].astype(np.float32)


def synthesize_broadband_noise(
    params: dict,
    n_samples: int,
    noise_gain: float,
    seed: int = 1234,
) -> np.ndarray:
    noise_mag_frames = np.asarray(
        params.get("noise_mag_frames", []), dtype=np.float32
    )
    noise_energy_frames = np.asarray(
        params.get("noise_energy_frames", []), dtype=np.float32
    )

    if noise_mag_frames.size == 0 or noise_energy_frames.size == 0 or noise_gain <= 0.0:
        return np.zeros(n_samples, dtype=np.float32)

    n_frames = noise_mag_frames.shape[0]
    hop = n_samples // n_frames

    if hop < 8:
        return np.zeros(n_samples, dtype=np.float32)

    impulse = amp_to_impulse_response(noise_mag_frames, hop)

    rng = np.random.default_rng(seed + 1)
    broadband_noise = rng.standard_normal(hop * n_frames).astype(np.float32)
    broadband_noise = broadband_noise - np.mean(broadband_noise)
    noise_blocks = broadband_noise.reshape(n_frames, hop)

    filtered_blocks = fft_convolve(noise_blocks, impulse)
    block_rms = np.sqrt(np.mean(filtered_blocks**2, axis=-1, keepdims=True) + 1e-8)
    filtered_blocks = filtered_blocks / block_rms

    filtered_noise = filtered_blocks.reshape(-1)
    if filtered_noise.shape[0] < n_samples:
        filtered_noise = np.pad(filtered_noise, (0, n_samples - filtered_noise.shape[0]))
    else:
        filtered_noise = filtered_noise[:n_samples]

    noise_energy = upsample_1d(noise_energy_frames, n_samples)

    return (noise_gain * noise_energy * filtered_noise).astype(np.float32)


def load_source_audio(path: Path, sr: int, seconds: float) -> np.ndarray:
    """
    Mirrors train_timbre.py's load_audio (same resample/trim-or-pad/peak
    normalize) so a vocoder-shifted note matches the duration and level of
    the DDSP-rendered notes it sits alongside. Duplicated here rather than
    imported so a plain render (no vocoder shifting needed) doesn't pull in
    train_timbre.py's much heavier torch/matplotlib imports.
    """
    y, _ = librosa.load(str(path), sr=sr, mono=True)

    max_len = int(sr * seconds)
    if len(y) > max_len:
        y = y[:max_len]
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak * 0.8

    return y.astype(np.float32)


MIN_VOICED_FRACTION = 0.5


def vocoder_is_reliable(source_audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Check whether WORLD's own pitch tracker (dio/stonemask) can find a
    consistent periodic pitch in this recording before trusting it to
    pitch-shift the whole note range. Some instruments (low brass in
    particular) have a fundamental that's weak relative to their upper
    harmonics -- a "missing fundamental" that confuses WORLD's tracker
    (built for voice/singing, where the fundamental is normally strong)
    and produces unrecognizable, warbling output even on notes close to
    the source pitch. Returns (reliable, voiced_fraction) for logging.
    """
    x = source_audio.astype(np.float64)
    f0_raw, t = pw.dio(x, sr)
    f0 = pw.stonemask(x, f0_raw, t, sr)
    voiced_fraction = float(np.mean(f0 > 0)) if len(f0) else 0.0
    return voiced_fraction >= MIN_VOICED_FRACTION, voiced_fraction


def vocoder_pitch_shift(
    source_audio: np.ndarray, sr: int, source_f0: float, target_f0: float
) -> np.ndarray:
    """
    Pitch-shift the recording to target_f0 via WORLD analysis-resynthesis
    (F0 + spectral envelope + aperiodicity) instead of approximating the
    sound from learned DDSP parameters. Only F0 is rescaled, so the
    instrument's formants/timbre stay close to the real recording. See
    VOCODER_MAX_SEMITONES for where this hands off to DDSP extrapolation.
    """
    x = source_audio.astype(np.float64)

    f0_raw, t = pw.dio(x, sr)
    f0 = pw.stonemask(x, f0_raw, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    ratio = target_f0 / source_f0
    shifted_f0 = f0 * ratio

    y = pw.synthesize(shifted_f0, sp, ap, sr).astype(np.float32)

    # WORLD's resynthesis doesn't guarantee the output sample count matches
    # the input exactly -- trim/pad back to the source length.
    target_len = len(source_audio)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak * 0.8

    return y


def synthesize_from_learned_timbre(
    params: dict,
    new_f0: float,
    sr: int,
    seconds: float,
    transient_gain: float = 0.15,
    output_gain: float = 1.0,
    noise_gain: float = 0.0,
    seed: int = 1234,
    use_frequency_mapping: bool = True,
    inharmonicity: float = 0.0,
    unison_voices: int = 1,
    unison_detune_cents: float = 0.0,
) -> np.ndarray:
    n_samples = int(sr * seconds)

    # These three define the actual learned sound -- there's no sensible
    # default to silently fall back to, so fail with a clear message
    # (e.g. a hand-edited or corrupted saved-timbre learned_params.json)
    # instead of a bare KeyError.
    for required_key in ("f0_hz", "amp_frames", "harmonic_distribution_frames"):
        if required_key not in params:
            raise KeyError(
                f"learned_params.json is missing required key {required_key!r} "
                "-- this timbre's saved parameters are incomplete or corrupted."
            )

    original_f0 = float(params["f0_hz"])

    amp_frames = np.asarray(params["amp_frames"], dtype=np.float32)

    original_harmonic_frames = np.asarray(
        params["harmonic_distribution_frames"],
        dtype=np.float32,
    )

    transient_frames = np.asarray(
        params.get("transient_env_frames", np.zeros_like(amp_frames)),
        dtype=np.float32,
    )

    if use_frequency_mapping:
        harmonic_frames = frequency_mapped_harmonic_frames(
            original_frames=original_harmonic_frames,
            original_f0=original_f0,
            new_f0=new_f0,
            sr=sr,
        )
    else:
        n_harmonics = original_harmonic_frames.shape[1]
        nyquist = sr / 2.0
        max_harmonic = int((nyquist * 0.95) // new_f0)
        effective_harmonics = max(1, min(n_harmonics, max_harmonic))
        harmonic_frames = original_harmonic_frames[:, :effective_harmonics]
        harmonic_frames = harmonic_frames / (
            np.sum(harmonic_frames, axis=1, keepdims=True) + 1e-8
        )

    effective_harmonics = harmonic_frames.shape[1]

    amp = upsample_1d(amp_frames, n_samples)
    harmonic_distribution = upsample_2d_time(harmonic_frames, n_samples)
    transient_env = upsample_1d(transient_frames, n_samples)

    t = np.arange(n_samples, dtype=np.float32) / sr
    harmonic_numbers = np.arange(1, effective_harmonics + 1, dtype=np.float32)

    # Stiff-string inharmonicity: f_n = n*f0*sqrt(1+B*n^2). B=0 is a no-op.
    inharmonic_multiplier = np.sqrt(
        1.0 + inharmonicity * harmonic_numbers**2
    ).astype(np.float32)

    # Multiple detuned unison voices (e.g. a piano note's 2-3 physical
    # strings). unison_voices=1 is a no-op.
    if unison_voices <= 1:
        detune_offsets_cents = np.array([0.0], dtype=np.float32)
    else:
        detune_offsets_cents = np.linspace(
            -unison_detune_cents, unison_detune_cents, unison_voices
        ).astype(np.float32)

    harmonic_audio = np.zeros(n_samples, dtype=np.float32)
    for cents in detune_offsets_cents:
        voice_f0 = new_f0 * (2.0 ** (cents / 1200.0))
        voice_freqs = voice_f0 * harmonic_numbers * inharmonic_multiplier
        phase = 2.0 * np.pi * voice_freqs[:, None] * t[None, :]
        sine_bank = np.sin(phase).T.astype(np.float32)
        harmonic_audio += np.sum(harmonic_distribution * sine_bank, axis=1)
    harmonic_audio /= len(detune_offsets_cents)

    audio = output_gain * amp * harmonic_audio

    rng = np.random.default_rng(seed)
    fixed_noise = rng.normal(0.0, 0.02, size=n_samples).astype(np.float32)
    fixed_noise = fixed_noise - np.mean(fixed_noise)

    audio = audio + transient_gain * transient_env * fixed_noise

    broadband_noise_audio = synthesize_broadband_noise(
        params, n_samples, noise_gain, seed=seed
    )
    audio = audio + broadband_noise_audio

    peak = np.max(np.abs(audio)) + 1e-8
    audio = audio / peak * 0.8

    return audio.astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove DC and fade edges
    audio = remove_dc(audio)
    audio = fade_edges(audio, sr)

    # Convert from 16 kHz to 44.1 kHz for Teensy playback
    target_sr = 44100
    audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=target_sr,
    )

    # Save as 16-bit PCM WAV
    sf.write(
        path,
        audio.astype(np.float32),
        target_sr,
        subtype="PCM_16",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--params", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="rendered_timbre")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=4.0)

    parser.add_argument("--f0", type=float, nargs="*", default=None)
    parser.add_argument("--midi", type=int, nargs="*", default=[32, 38, 44, 50, 56, 60])

    parser.add_argument(
        "--transient_gain",
        type=float,
        default=None,
        help=(
            "Override the transient noise gain. If omitted, uses the value "
            "actually learned during training (params['transient_gain']), "
            "matching the mix balance heard in Play Output."
        ),
    )

    parser.add_argument(
        "--noise_gain",
        type=float,
        default=None,
        help=(
            "Override the continuous filtered-noise gain. If omitted, uses "
            "the value actually learned during training (params['noise_gain'])."
        ),
    )

    parser.add_argument(
        "--old_mapping",
        action="store_true",
        help="Use old harmonic-index mapping instead of pitch-aware frequency mapping.",
    )

    parser.add_argument(
        "--source_audio",
        type=str,
        default=None,
        help=(
            "Path to the original recording this timbre was learned from. "
            "When given, notes within --vocoder_max_semitones of the "
            "learned f0 are generated by WORLD-vocoder pitch-shifting this "
            "recording (reuses the real timbre) instead of DDSP "
            "resynthesis. Notes further out, or all notes if this is "
            "omitted, use DDSP as before."
        ),
    )
    parser.add_argument(
        "--vocoder_max_semitones",
        type=float,
        default=VOCODER_MAX_SEMITONES,
        help="Max distance from the source pitch (semitones) that still uses vocoder shifting.",
    )

    parser.add_argument(
        "--inharmonicity",
        type=float,
        default=0.0,
        help="Stiff-string inharmonicity coefficient B (f_n = n*f0*sqrt(1+B*n^2)). 0 = exact harmonics (old behavior).",
    )
    parser.add_argument(
        "--unison_voices",
        type=int,
        default=1,
        help="Number of detuned unison voices to sum per note (e.g. piano's multiple strings per note). 1 = old single-voice behavior.",
    )
    parser.add_argument(
        "--unison_detune_cents",
        type=float,
        default=0.0,
        help="Total spread (cents) across unison_voices, symmetric around the target pitch.",
    )

    args = parser.parse_args()

    params_path = Path(args.params)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(params_path, "r") as f:
        params = json.load(f)

    output_gain = float(params.get("output_gain", 1.0))
    transient_gain = (
        args.transient_gain
        if args.transient_gain is not None
        else float(params.get("transient_gain", 0.15))
    )
    noise_gain = (
        args.noise_gain
        if args.noise_gain is not None
        else float(params.get("noise_gain", 0.0))
    )
    print(f"output_gain: {output_gain:.4f} (learned)")
    print(
        f"transient_gain: {transient_gain:.4f} "
        f"({'learned' if args.transient_gain is None else 'CLI override'})"
    )
    print(
        f"noise_gain: {noise_gain:.4f} "
        f"({'learned' if args.noise_gain is None else 'CLI override'})"
    )

    if args.f0 is not None and len(args.f0) > 0:
        render_items = [(f"f0_{freq:.2f}Hz", float(freq)) for freq in args.f0]
    else:
        render_items = [
            (f"midi_{note}_{midi_to_hz(note):.2f}Hz", midi_to_hz(note))
            for note in args.midi
        ]

    original_f0 = params.get("f0_hz")
    original_f0 = float(original_f0) if original_f0 else None

    source_audio = None
    voiced_fraction = None
    if args.source_audio:
        source_path = Path(args.source_audio)
        if source_path.exists() and original_f0:
            source_audio = load_source_audio(source_path, sr=args.sr, seconds=args.seconds)
            reliable, voiced_fraction = vocoder_is_reliable(source_audio, args.sr)
            if not reliable:
                print(
                    f"Vocoder pitch tracking unreliable on this recording "
                    f"(WORLD found a consistent pitch in only "
                    f"{voiced_fraction*100:.0f}% of frames, need >= "
                    f"{MIN_VOICED_FRACTION*100:.0f}%) -- disabling vocoder "
                    "for this render, using DDSP for all notes instead."
                )
                source_audio = None

    print("Loaded learned timbre parameters from:")
    print(params_path)
    print(f"Original learned f0: {params.get('f0_hz', 'unknown')} Hz")
    print(f"Pitch-aware frequency mapping: {not args.old_mapping}")
    print(
        f"Transparent de-click: fade-in {FADE_IN_SECONDS}s, fade-out {FADE_OUT_SECONDS}s"
    )
    if source_audio is not None:
        print(
            f"Vocoder pitch-shift hybrid: enabled (source={args.source_audio}, "
            f"voiced_fraction={voiced_fraction*100:.0f}%, "
            f"max {args.vocoder_max_semitones:.1f} semitones from source)"
        )
    else:
        print("Vocoder pitch-shift hybrid: disabled (no usable --source_audio, or unreliable pitch tracking), using DDSP for all notes")
    print(f"Rendering {len(render_items)} notes...")

    summary = {
        "source_params": str(params_path),
        "original_f0_hz": params.get("f0_hz", None),
        "pitch_aware_frequency_mapping": not args.old_mapping,
        "vocoder_hybrid_enabled": source_audio is not None,
        "vocoder_max_semitones": args.vocoder_max_semitones,
        "transparent_declick": {
            "fade_in_seconds": FADE_IN_SECONDS,
            "fade_out_seconds": FADE_OUT_SECONDS,
        },
        "rendered": [],
    }

    for name, new_f0 in render_items:
        semitone_distance = (
            abs(12.0 * np.log2(new_f0 / original_f0)) if original_f0 else None
        )
        use_vocoder = (
            source_audio is not None
            and semitone_distance is not None
            and semitone_distance <= args.vocoder_max_semitones
        )

        if use_vocoder:
            audio = vocoder_pitch_shift(
                source_audio=source_audio,
                sr=args.sr,
                source_f0=original_f0,
                target_f0=new_f0,
            )
            method = "vocoder"
        else:
            audio = synthesize_from_learned_timbre(
                params=params,
                new_f0=new_f0,
                sr=args.sr,
                seconds=args.seconds,
                transient_gain=transient_gain,
                output_gain=output_gain,
                noise_gain=noise_gain,
                use_frequency_mapping=not args.old_mapping,
                inharmonicity=args.inharmonicity,
                unison_voices=args.unison_voices,
                unison_detune_cents=args.unison_detune_cents,
            )
            method = "ddsp"

        wav_path = out_dir / f"{name}.wav"
        save_wav(wav_path, audio, args.sr)

        summary["rendered"].append(
            {
                "name": name,
                "f0_hz": new_f0,
                "file": str(wav_path),
                "method": method,
                "semitones_from_source": semitone_distance,
            }
        )

        distance_note = f", {semitone_distance:.1f} semitones from source" if semitone_distance is not None else ""
        print(f"Saved {wav_path} ({method}{distance_note})")

    with open(out_dir / "render_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Rendered files saved in: {out_dir}")


if __name__ == "__main__":
    main()
