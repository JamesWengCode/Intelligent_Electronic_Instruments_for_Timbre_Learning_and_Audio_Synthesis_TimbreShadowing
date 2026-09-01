"""
predict_instrument.py

Predict one audio file's instrument AND family (classes come from
models/class_names.json / models/family_names.json, written by
train_classifier.py).

Run from project root:

python src/predict_instrument.py --audio input/target.wav

Output example:

instrument:violin
instrument_confidence=0.9823
family:strings
family_confidence=0.9910
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from train_classifier import HierarchicalInstrumentCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "instrument_cnn_hierarchical.pth"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
DEFAULT_FAMILY_NAMES_PATH = PROJECT_ROOT / "models" / "family_names.json"

SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


def audio_to_logmel(path: Path) -> torch.Tensor:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    target_len = int(SAMPLE_RATE * DURATION_SECONDS)

    if len(y) > target_len:
        y = y[:target_len]

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)

    x = torch.tensor(logmel, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=str, default=str(PROJECT_ROOT / "input" / "target.wav")
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--class_names", type=str, default=str(DEFAULT_CLASS_NAMES_PATH)
    )
    parser.add_argument(
        "--family_names", type=str, default=str(DEFAULT_FAMILY_NAMES_PATH)
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    class_names_path = Path(args.class_names)
    family_names_path = Path(args.family_names)

    if not audio_path.exists():
        raise FileNotFoundError(f"Cannot find audio file: {audio_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model file: {model_path}")

    if not class_names_path.exists():
        raise FileNotFoundError(f"Cannot find class names file: {class_names_path}")

    if not family_names_path.exists():
        raise FileNotFoundError(f"Cannot find family names file: {family_names_path}")

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open(family_names_path, "r", encoding="utf-8") as f:
        family_names = json.load(f)

    device = torch.device("cpu")

    model = HierarchicalInstrumentCNN(
        num_instruments=len(class_names),
        num_families=len(family_names),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = audio_to_logmel(audio_path).to(device)

    with torch.no_grad():
        family_logits, instrument_logits = model(x)
        instrument_probs = torch.softmax(instrument_logits, dim=1)[0]
        family_probs = torch.softmax(family_logits, dim=1)[0]
        instrument_idx = int(torch.argmax(instrument_probs).item())
        family_idx = int(torch.argmax(family_probs).item())

    predicted_instrument = class_names[instrument_idx]
    instrument_confidence = float(instrument_probs[instrument_idx].item())
    predicted_family = family_names[family_idx]
    family_confidence = float(family_probs[family_idx].item())

    print(f"instrument:{predicted_instrument}")
    print(f"instrument_confidence={instrument_confidence:.4f}")
    print(f"family:{predicted_family}")
    print(f"family_confidence={family_confidence:.4f}")


if __name__ == "__main__":
    main()
