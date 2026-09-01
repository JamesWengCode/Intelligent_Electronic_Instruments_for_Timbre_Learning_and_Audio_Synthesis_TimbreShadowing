"""
enhance_rendered_notes.py

Post-process rendered notes.

Currently: adds a short high-frequency pluck/click layer to the start of
each note, for electric_bass.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


def bandpass_noise(noise, sr, low=700, high=5000):
    nyq = sr / 2
    high = min(high, nyq * 0.95)
    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, noise)


def add_bass_click(path: Path, amount: float = 0.45):
    y, sr = sf.read(path)

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)

    n = len(y)
    click_len = min(int(sr * 0.055), n)  # 55 ms
    snap_len = min(int(sr * 0.008), n)   # 8 ms

    rng = np.random.default_rng(42)

    noise = rng.standard_normal(click_len).astype(np.float32)
    noise = bandpass_noise(noise, sr, low=900, high=5500)

    env = np.exp(-np.linspace(0, 7.0, click_len)).astype(np.float32)
    click = noise * env

    # Very short initial finger snap.
    snap = np.zeros(click_len, dtype=np.float32)
    snap[:snap_len] = rng.standard_normal(snap_len).astype(np.float32)
    snap[:snap_len] *= np.exp(-np.linspace(0, 8.0, snap_len)).astype(np.float32)

    click = click + 0.6 * snap

    click = click / (np.max(np.abs(click)) + 1e-8)

    # Scale by current audio peak.
    peak = np.max(np.abs(y)) + 1e-8
    click = click * peak * amount

    out = y.copy()
    out[:click_len] += click

    out = out / (np.max(np.abs(out)) + 1e-8) * 0.95

    sf.write(path, out, sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--amount", type=float, default=0.45)
    parser.add_argument("--include_output", type=str, default="")
    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.exists():
        raise FileNotFoundError(folder)

    wavs = sorted(folder.glob("*.wav"))

    for wav in wavs:
        add_bass_click(wav, amount=args.amount)
        print(f"Enhanced bass click: {wav}")

    if args.include_output:
        output_path = Path(args.include_output)
        if output_path.exists():
            add_bass_click(output_path, amount=args.amount)
            print(f"Enhanced bass click: {output_path}")


if __name__ == "__main__":
    main()
