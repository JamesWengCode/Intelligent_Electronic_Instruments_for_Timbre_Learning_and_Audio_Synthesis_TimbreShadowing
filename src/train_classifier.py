"""
train_classifier.py

Train a hierarchical, multi-head instrument classifier: a shared CNN trunk
feeding two heads -- a coarse Family head (strings/woodwinds/brass/keyboard/
guitar) and a fine-grained Instrument head (violin, cello, ...). Family
labels are derived automatically from instrument labels via
src/instrument_taxonomy.py -- no separate family-level data needed.

Instrument classes are auto-discovered from data/<family>/<instrument>/
folders. The family comes from the parent folder name (must be one of
instrument_taxonomy.FAMILY_NAMES); the instrument folder name is
cross-checked against instrument_taxonomy.INSTRUMENT_TO_FAMILY so a
misplaced folder is caught instead of silently mislabeled. Add a new
instrument by adding it to instrument_taxonomy.py, then dropping a
data/<family>/<instrument_name>/ folder with its audio -- no other code
change needed. Instruments without a data folder yet are simply skipped
until their audio is added.

Expected project structure:

Capstone_Software_System/
├── data/
│   ├── strings/
│   │   ├── violin/
│   │   └── cello/
│   ├── guitar/
│   │   └── electric_bass/
│   └── ...
├── models/
└── src/
    ├── instrument_taxonomy.py
    └── train_classifier.py

Run from project root:

python src/train_classifier.py

Output:

models/instrument_cnn_hierarchical.pth
models/class_names.json      (instrument names, in label-index order)
models/family_names.json     (family names, in label-index order)
models/class_ranges.json     (per-instrument MIDI range)
"""

import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from instrument_taxonomy import FAMILY_NAMES, INSTRUMENT_TO_FAMILY, family_of

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

AUDIO_EXTENSIONS = (".wav", ".aif", ".aiff", ".flac", ".mp3")

SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
# Three-way split: VAL is used during training to pick the best checkpoint,
# so accuracy measured on it is optimistically biased (the model was
# selected to do well on exactly that set). TEST is never touched until the
# very end -- its accuracy is the honest number.
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

FAMILY_LOSS_WEIGHT = 0.3  # auxiliary task -- instrument head is the target

AUGMENT_TRAIN = True

GAIN_RANGE = (0.7, 1.3)
NOISE_PROB = 0.5
NOISE_LEVEL_RANGE = (0.001, 0.01)
TIME_SHIFT_SECONDS = 0.05

PITCH_SHIFT_PROB = 0.5
PITCH_SHIFT_SEMITONES = (-2.0, 2.0)
TIME_STRETCH_PROB = 0.5
TIME_STRETCH_RATE_RANGE = (0.85, 1.15)

SCARCE_CLASS_MAX_FILES = 100

FREQ_MASK_WIDTH = 12
NUM_FREQ_MASKS = 2
TIME_MASK_WIDTH = 20
NUM_TIME_MASKS = 2


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_waveform(y: np.ndarray, heavy: bool = False) -> np.ndarray:
    # Pitch-shift/time-stretch are expensive, so only applied to classes
    # too small to rely on natural variety (see SCARCE_CLASS_MAX_FILES).
    if heavy:
        if np.random.rand() < PITCH_SHIFT_PROB:
            n_steps = np.random.uniform(*PITCH_SHIFT_SEMITONES)
            y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)

        if np.random.rand() < TIME_STRETCH_PROB:
            rate = np.random.uniform(*TIME_STRETCH_RATE_RANGE)
            y = librosa.effects.time_stretch(y, rate=rate)

    y = y * np.random.uniform(*GAIN_RANGE)

    if np.random.rand() < NOISE_PROB:
        noise_level = np.random.uniform(*NOISE_LEVEL_RANGE)
        y = y + np.random.randn(len(y)).astype(np.float32) * noise_level

    max_shift = int(SAMPLE_RATE * TIME_SHIFT_SECONDS)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift != 0:
        y = np.roll(y, shift)

    return y.astype(np.float32)


def spec_augment(logmel: np.ndarray) -> np.ndarray:
    """
    Random frequency/time masking (SpecAugment). Mask value is the clip's own
    min so masked bands blend into the already-normalized background instead
    of standing out as an artificial edge.
    """
    logmel = logmel.copy()
    n_mels, n_frames = logmel.shape
    mask_value = logmel.min()

    for _ in range(NUM_FREQ_MASKS):
        width = np.random.randint(0, FREQ_MASK_WIDTH + 1)
        start = np.random.randint(0, max(1, n_mels - width))
        logmel[start : start + width, :] = mask_value

    for _ in range(NUM_TIME_MASKS):
        width = np.random.randint(0, TIME_MASK_WIDTH + 1)
        start = np.random.randint(0, max(1, n_frames - width))
        logmel[:, start : start + width] = mask_value

    return logmel


def audio_to_logmel(
    path: Path, augment: bool = False, heavy_augment: bool = False
) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    target_len = int(SAMPLE_RATE * DURATION_SECONDS)

    if len(y) > target_len:
        y = y[:target_len]

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    if augment:
        y = augment_waveform(y, heavy=heavy_augment)
        # time_stretch changes duration; re-fix length so every item in a
        # batch produces the same spectrogram shape.
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

    if augment:
        logmel = spec_augment(logmel)

    return logmel.astype(np.float32)


class AudioDataset(Dataset):
    def __init__(self, items, augment: bool = False, heavy_augment_labels=None):
        self.items = items  # (path, instrument_label, family_label)
        self.augment = augment
        self.heavy_augment_labels = heavy_augment_labels or set()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, instrument_label, family_label = self.items[idx]
        heavy = instrument_label in self.heavy_augment_labels
        x = audio_to_logmel(path, augment=self.augment, heavy_augment=heavy)
        x = torch.tensor(x).unsqueeze(0)
        return (
            x,
            torch.tensor(instrument_label, dtype=torch.long),
            torch.tensor(family_label, dtype=torch.long),
        )


class HierarchicalInstrumentCNN(nn.Module):
    """
    Shared conv trunk feeding two independent heads. The family head is an
    auxiliary task -- it doesn't gate or condition the instrument head --
    but sharing the trunk means the instrument head benefits from features
    that are also useful for coarse family discrimination.
    """

    def __init__(self, num_instruments: int, num_families: int):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc_shared = nn.Linear(64 * 4 * 4, 128)

        self.family_head = nn.Linear(128, num_families)
        self.instrument_head = nn.Linear(128, num_instruments)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        shared = self.dropout(F.relu(self.fc_shared(x)))

        family_logits = self.family_head(shared)
        instrument_logits = self.instrument_head(shared)

        return family_logits, instrument_logits


PITCH_RANGE_MAX_SAMPLES = 40
PITCH_RANGE_FMIN = 40.0
PITCH_RANGE_FMAX = 2000.0
PITCH_RANGE_FIXED_COUNT = 36  # every instrument gets the same note count
# Floor for the computed window, snapped to a C. Without this, a window
# centered on a low instrument's median pitch can dip into a range no real
# instrument reaches (e.g. electric_bass pushing down to C0, ~16Hz) -- C1
# (~33Hz) sits below anything in the taxonomy needs.
PITCH_RANGE_MIN_START_MIDI = 24  # C1


def estimate_class_pitch_range(audio_paths):
    """
    Estimate a playable MIDI range for a class from its own audio (sampling
    up to PITCH_RANGE_MAX_SAMPLES files), instead of a hand-picked range.
    Fits a fixed-size window starting on a C around the median detected f0.

    Returns (start_midi, count) or None if no pitch could be detected.
    """
    paths = list(audio_paths)
    if len(paths) > PITCH_RANGE_MAX_SAMPLES:
        rng = random.Random(RANDOM_SEED)
        paths = rng.sample(paths, PITCH_RANGE_MAX_SAMPLES)

    midi_notes = []
    for path in paths:
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue

        if len(y) < SAMPLE_RATE * 0.2:
            continue

        f0 = librosa.yin(
            y, fmin=PITCH_RANGE_FMIN, fmax=PITCH_RANGE_FMAX, sr=SAMPLE_RATE
        )
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > PITCH_RANGE_FMIN) & (f0 < PITCH_RANGE_FMAX)]

        if len(f0) == 0:
            continue

        median_f0 = float(np.median(f0))
        midi_notes.append(69.0 + 12.0 * np.log2(median_f0 / 440.0))

    if not midi_notes:
        return None

    median_midi = float(np.median(midi_notes))
    # Snap the window's start to the nearest C, not its center -- rounding
    # the center and then subtracting half the window doesn't land on a C.
    approx_start = median_midi - PITCH_RANGE_FIXED_COUNT / 2.0
    start = int(round(approx_start / 12.0)) * 12
    start = max(start, PITCH_RANGE_MIN_START_MIDI)

    return start, PITCH_RANGE_FIXED_COUNT


def discover_classes() -> list:
    """
    Auto-discover instrument classes from data/<family>/<instrument>/
    folders. The family is taken directly from the parent folder name (must
    be one of instrument_taxonomy.FAMILY_NAMES); the instrument folder name
    is cross-checked against instrument_taxonomy.INSTRUMENT_TO_FAMILY so a
    misplaced folder (e.g. "violin" accidentally dropped under brass/) is
    caught instead of silently mislabeled.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    classes = []

    for family_dir in sorted(DATA_DIR.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("."):
            continue

        family_name = family_dir.name
        if family_name not in FAMILY_NAMES:
            print(
                f"Skipping data/{family_name}/ -- not a known family "
                f"(expected one of {FAMILY_NAMES})"
            )
            continue

        for instrument_dir in sorted(family_dir.iterdir()):
            if not instrument_dir.is_dir() or instrument_dir.name.startswith("."):
                continue

            instrument_name = instrument_dir.name
            expected_family = INSTRUMENT_TO_FAMILY.get(instrument_name)

            if expected_family is None:
                print(
                    f"Skipping data/{family_name}/{instrument_name}/ -- not "
                    "in instrument_taxonomy.INSTRUMENT_TO_FAMILY"
                )
                continue

            if expected_family != family_name:
                print(
                    f"Skipping data/{family_name}/{instrument_name}/ -- "
                    f"taxonomy says '{instrument_name}' belongs under "
                    f"{expected_family}/, not {family_name}/"
                )
                continue

            classes.append(instrument_name)

    if not classes:
        raise RuntimeError(
            f"No recognized instrument folders found under: {DATA_DIR}\n"
            "Expected data/<family>/<instrument>/ matching instrument_taxonomy.py."
        )

    return classes


def collect_dataset(class_names: list):
    items = []

    for label_idx, class_name in enumerate(class_names):
        family_name = family_of(class_name)
        class_dir = DATA_DIR / family_name / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        audio_files = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )

        if len(audio_files) == 0:
            raise RuntimeError(f"No audio files found in: {class_dir}")

        print(f"{family_name}/{class_name}: {len(audio_files)} files")

        family_idx = FAMILY_NAMES.index(family_name)

        for audio_path in audio_files:
            items.append((audio_path, label_idx, family_idx))

    return items


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    instrument_correct = 0
    family_correct = 0
    total = 0

    for x, instrument_y, family_y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        instrument_y = instrument_y.to(device)
        family_y = family_y.to(device)

        optimizer.zero_grad()
        family_logits, instrument_logits = model(x)

        instrument_loss = criterion(instrument_logits, instrument_y)
        family_loss = criterion(family_logits, family_y)
        loss = instrument_loss + FAMILY_LOSS_WEIGHT * family_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        instrument_correct += (
            (torch.argmax(instrument_logits, dim=1) == instrument_y).sum().item()
        )
        family_correct += (
            (torch.argmax(family_logits, dim=1) == family_y).sum().item()
        )
        total += x.size(0)

    return total_loss / total, instrument_correct / total, family_correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    instrument_correct = 0
    family_correct = 0
    total = 0

    all_instrument_true = []
    all_instrument_pred = []
    all_family_true = []
    all_family_pred = []

    with torch.no_grad():
        for x, instrument_y, family_y in tqdm(loader, desc="Val", leave=False):
            x = x.to(device)
            instrument_y = instrument_y.to(device)
            family_y = family_y.to(device)

            family_logits, instrument_logits = model(x)

            instrument_loss = criterion(instrument_logits, instrument_y)
            family_loss = criterion(family_logits, family_y)
            loss = instrument_loss + FAMILY_LOSS_WEIGHT * family_loss

            total_loss += loss.item() * x.size(0)

            instrument_pred = torch.argmax(instrument_logits, dim=1)
            family_pred = torch.argmax(family_logits, dim=1)

            instrument_correct += (instrument_pred == instrument_y).sum().item()
            family_correct += (family_pred == family_y).sum().item()
            total += x.size(0)

            all_instrument_true.extend(instrument_y.cpu().numpy().tolist())
            all_instrument_pred.extend(instrument_pred.cpu().numpy().tolist())
            all_family_true.extend(family_y.cpu().numpy().tolist())
            all_family_pred.extend(family_pred.cpu().numpy().tolist())

    return {
        "loss": total_loss / total,
        "instrument_acc": instrument_correct / total,
        "family_acc": family_correct / total,
        "instrument_true": all_instrument_true,
        "instrument_pred": all_instrument_pred,
        "family_true": all_family_true,
        "family_pred": all_family_pred,
    }


def main():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes()
    families_present = sorted({family_of(c) for c in class_names})

    print("=" * 80)
    print(f"Training hierarchical instrument classifier")
    print(f"Instruments ({len(class_names)}): {class_names}")
    print(f"Families present ({len(families_present)}): {families_present}")
    print("=" * 80)

    items = collect_dataset(class_names)
    instrument_labels = [label for _, label, _ in items]
    family_labels = [label for _, _, label in items]

    print("\nEstimating playable MIDI range per class from audio...")
    class_ranges = {}
    for label_idx, class_name in enumerate(class_names):
        class_paths = [p for p, l, _ in items if l == label_idx]
        estimated = estimate_class_pitch_range(class_paths)
        if estimated is None:
            print(f"  {class_name}: no pitch detected, no range saved")
            continue
        start_midi, count = estimated
        class_ranges[class_name] = {"start_midi": start_midi, "count": count}
        print(
            f"  {class_name}: MIDI {start_midi}-{start_midi + count - 1} "
            f"({count} notes)"
        )

    with open(MODEL_DIR / "class_ranges.json", "w", encoding="utf-8") as f:
        json.dump(class_ranges, f, indent=2)

    counts = np.bincount(instrument_labels, minlength=len(class_names))
    tiny_classes = [class_names[i] for i, c in enumerate(counts) if c < 10]
    if tiny_classes:
        print(
            f"\nWarning: very few samples for {tiny_classes} -- "
            "val accuracy for these classes will not be statistically "
            "meaningful.\n"
        )

    scarce_labels = {i for i, c in enumerate(counts) if c < SCARCE_CLASS_MAX_FILES}
    if scarce_labels:
        print(
            "Heavy augmentation (pitch-shift/time-stretch) enabled for: "
            f"{[class_names[i] for i in scarce_labels]}\n"
        )

    # Split off the test set first (stratified on instrument label), then
    # split what's left into train/val. Test is held out from both training
    # and checkpoint selection -- it's only touched once, at the very end.
    train_val_items, test_items = train_test_split(
        items,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=instrument_labels,
    )
    train_val_instrument_labels = [label for _, label, _ in train_val_items]
    train_items, val_items = train_test_split(
        train_val_items,
        test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=train_val_instrument_labels,
    )

    print(f"\nTrain files: {len(train_items)}")
    print(f"Val files:   {len(val_items)}")
    print(f"Test files:  {len(test_items)}")

    train_loader = DataLoader(
        AudioDataset(
            train_items,
            augment=AUGMENT_TRAIN,
            heavy_augment_labels=scarce_labels,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        AudioDataset(val_items, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        AudioDataset(test_items, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = HierarchicalInstrumentCNN(
        num_instruments=len(class_names),
        num_families=len(FAMILY_NAMES),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = MODEL_DIR / "instrument_cnn_hierarchical.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_instrument_acc, train_family_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_result = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} "
            f"train_instr_acc={train_instrument_acc:.4f} "
            f"train_fam_acc={train_family_acc:.4f} | "
            f"val_loss={val_result['loss']:.4f} "
            f"val_instr_acc={val_result['instrument_acc']:.4f} "
            f"val_fam_acc={val_result['family_acc']:.4f}"
        )

        if val_result["instrument_acc"] > best_val_acc:
            best_val_acc = val_result["instrument_acc"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")

    class_json = MODEL_DIR / "class_names.json"
    with open(class_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    family_json = MODEL_DIR / "family_names.json"
    with open(family_json, "w", encoding="utf-8") as f:
        json.dump(FAMILY_NAMES, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))

    print("\n" + "=" * 80)
    print("Held-out TEST report (never seen during training or checkpoint selection)")
    print("=" * 80)

    test_result = evaluate(model, test_loader, criterion, device)

    print(f"Best instrument val acc (checkpoint selection, optimistic): {best_val_acc:.4f}")
    print(f"Test instrument acc (honest number): {test_result['instrument_acc']:.4f}")
    print(f"Test family acc (honest number): {test_result['family_acc']:.4f}")

    instrument_label_ids = list(range(len(class_names)))
    family_label_ids = list(range(len(FAMILY_NAMES)))

    print("\nInstrument classification report (test set):")
    instrument_report = classification_report(
        test_result["instrument_true"],
        test_result["instrument_pred"],
        labels=instrument_label_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            test_result["instrument_true"],
            test_result["instrument_pred"],
            labels=instrument_label_ids,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("\nInstrument confusion matrix (test set):")
    print(
        confusion_matrix(
            test_result["instrument_true"],
            test_result["instrument_pred"],
            labels=instrument_label_ids,
        )
    )

    print("\nFamily classification report (test set):")
    family_report = classification_report(
        test_result["family_true"],
        test_result["family_pred"],
        labels=family_label_ids,
        target_names=FAMILY_NAMES,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            test_result["family_true"],
            test_result["family_pred"],
            labels=family_label_ids,
            target_names=FAMILY_NAMES,
            zero_division=0,
        )
    )

    print("\nFamily confusion matrix (test set):")
    print(
        confusion_matrix(
            test_result["family_true"],
            test_result["family_pred"],
            labels=family_label_ids,
        )
    )

    # Per-instrument "accuracy" is recall: the fraction of that
    # instrument's held-out test samples the model got right. Overwritten
    # every run -- the GUI's Recognition Accuracy button reads this file.
    metrics = {
        "overall_accuracy": test_result["instrument_acc"],
        "per_instrument_accuracy": {
            name: instrument_report[name]["recall"] for name in class_names
        },
        "overall_family_accuracy": test_result["family_acc"],
        "per_family_accuracy": {
            name: family_report[name]["recall"] for name in FAMILY_NAMES
        },
    }
    metrics_json = MODEL_DIR / "training_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(best_path)
    print(class_json)
    print(family_json)
    print(MODEL_DIR / "class_ranges.json")
    print(metrics_json)


if __name__ == "__main__":
    main()
