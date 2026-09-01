"""Permanent user-managed sound library for learned DDSP timbres.

Mirrors data/'s <family>/<instrument>/ layout exactly (see
instrument_taxonomy.py): family and instrument folder names always come
from the shared taxonomy, derived from a sound's own detected instrument
class. There is no free-form category system -- a sound's location is
fully determined by its classification, so saved_timbres/ can never drift
out of sync with data/.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from instrument_taxonomy import FAMILY_NAMES, INSTRUMENT_TO_FAMILY, family_of

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOUND_LIBRARY_ROOT = PROJECT_ROOT / "saved_timbres"


def _slug(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("Name cannot be empty.")
    return value


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def display_name(taxonomy_folder_name: str) -> str:
    return taxonomy_folder_name.replace("_", " ").title()


def _instrument_folder(instrument_class: str) -> Path:
    if instrument_class not in INSTRUMENT_TO_FAMILY:
        raise ValueError(
            f"Unknown instrument '{instrument_class}' -- add it to "
            "src/instrument_taxonomy.py first."
        )
    folder = SOUND_LIBRARY_ROOT / family_of(instrument_class) / instrument_class
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_taxonomy() -> list[dict]:
    """
    Every family with its instruments, in display order, each with its
    current saved-sound count. Families/instruments are always listed even
    with zero saved sounds -- the taxonomy, not the filesystem, defines
    what exists.
    """
    families = []
    for family in FAMILY_NAMES:
        instruments = []
        for instrument, inst_family in INSTRUMENT_TO_FAMILY.items():
            if inst_family != family:
                continue
            folder = SOUND_LIBRARY_ROOT / family / instrument
            count = 0
            if folder.is_dir():
                count = sum(
                    1
                    for p in folder.iterdir()
                    if p.is_dir() and (p / "metadata.json").exists()
                )
            instruments.append(
                {
                    "name": display_name(instrument),
                    "instrument_folder": instrument,
                    "family_folder": family,
                    "sound_count": count,
                }
            )
        instruments.sort(key=lambda item: item["name"].lower())
        families.append(
            {
                "name": display_name(family),
                "family_folder": family,
                "instruments": instruments,
            }
        )
    return families


def save_sound(
    sound_name: str,
    detected_instrument_class: str,
    learned_params_path: Path,
    preview_wav_path: Path | None = None,
    source_audio_path: Path | None = None,
    last_correction_file: Path | None = None,
) -> Path:
    name = sound_name.strip()
    if not name:
        raise ValueError("Sound name cannot be empty.")
    if not learned_params_path.exists():
        raise FileNotFoundError(f"Missing learned parameters: {learned_params_path}")

    instrument_folder = _instrument_folder(detected_instrument_class)

    sound_folder = instrument_folder / _slug(name)
    if sound_folder.exists():
        raise FileExistsError(
            f"A sound named '{name}' already exists under "
            f"{display_name(detected_instrument_class)}."
        )

    sound_folder.mkdir(parents=True)
    try:
        shutil.copy2(learned_params_path, sound_folder / "learned_params.json")
        if preview_wav_path and preview_wav_path.exists():
            shutil.copy2(preview_wav_path, sound_folder / "preview.wav")
        if source_audio_path and source_audio_path.exists():
            shutil.copy2(source_audio_path, sound_folder / "source_audio.wav")

        metadata = {
            "name": name,
            "folder_name": sound_folder.name,
            "family_folder": instrument_folder.parent.name,
            "instrument_folder": instrument_folder.name,
            "detected_instrument_class": detected_instrument_class,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        # Same field correct_sound_class writes -- lets delete_sound also
        # clean up a correction filed before this was saved.
        if last_correction_file is not None:
            metadata["last_correction_file"] = str(last_correction_file)
        _write_json(sound_folder / "metadata.json", metadata)
    except Exception:
        shutil.rmtree(sound_folder, ignore_errors=True)
        raise

    return sound_folder


def _sound_record(family_folder: str, instrument_folder: str, sound_folder: Path) -> dict | None:
    params = sound_folder / "learned_params.json"
    metadata_path = sound_folder / "metadata.json"
    if not sound_folder.is_dir() or not params.exists() or not metadata_path.exists():
        return None

    try:
        metadata = _read_json(metadata_path)
    except (OSError, json.JSONDecodeError, TypeError):
        return None

    return {
        **metadata,
        "name": metadata.get("name", sound_folder.name),
        "folder_name": sound_folder.name,
        "family_folder": family_folder,
        "instrument_folder": instrument_folder,
        "detected_instrument_class": metadata.get(
            "detected_instrument_class", instrument_folder
        ),
        "folder_path": str(sound_folder),
        "params_path": str(params),
        "preview_path": (
            str(sound_folder / "preview.wav")
            if (sound_folder / "preview.wav").exists()
            else None
        ),
        "source_audio_path": (
            str(sound_folder / "source_audio.wav")
            if (sound_folder / "source_audio.wav").exists()
            else None
        ),
    }


def list_sounds(
    family_folder: str | None = None, instrument_folder: str | None = None
) -> list[dict]:
    """List saved sounds, optionally filtered to one family and/or instrument."""
    if instrument_folder is not None and family_folder is None:
        raise ValueError("instrument_folder requires family_folder.")

    families = [family_folder] if family_folder is not None else FAMILY_NAMES

    results = []
    for family in families:
        if instrument_folder is not None:
            instruments = [instrument_folder]
        else:
            instruments = [
                name for name, fam in INSTRUMENT_TO_FAMILY.items() if fam == family
            ]
        for instrument in instruments:
            folder = SOUND_LIBRARY_ROOT / family / instrument
            if not folder.is_dir():
                continue
            for sound_folder in sorted(folder.iterdir()):
                record = _sound_record(family, instrument, sound_folder)
                if record is not None:
                    results.append(record)
    return sorted(results, key=lambda item: item["name"].lower())


def load_sound(family_folder: str, instrument_folder: str, sound_folder_name: str) -> dict:
    sound = SOUND_LIBRARY_ROOT / family_folder / instrument_folder / sound_folder_name
    record = _sound_record(family_folder, instrument_folder, sound)
    if record is None:
        raise FileNotFoundError(f"Sound not found or incomplete: {sound}")
    return record


def rename_sound(
    family_folder: str, instrument_folder: str, sound_folder_name: str, new_name: str
) -> Path:
    folder = SOUND_LIBRARY_ROOT / family_folder / instrument_folder
    source = folder / sound_folder_name
    if not source.is_dir():
        raise FileNotFoundError(f"Sound not found: {source}")

    display = new_name.strip()
    if not display:
        raise ValueError("Sound name cannot be empty.")

    destination = folder / _slug(display)
    if destination != source and destination.exists():
        raise FileExistsError(f"A sound named '{display}' already exists.")

    if destination != source:
        source.rename(destination)

    # Roll back the rename if metadata.json update fails, so folder_name
    # never permanently disagrees with the actual path.
    try:
        metadata_path = destination / "metadata.json"
        metadata = _read_json(metadata_path)
        metadata["name"] = display
        metadata["folder_name"] = destination.name
        _write_json(metadata_path, metadata)
    except Exception:
        if destination != source and destination.exists():
            destination.rename(source)
        raise

    return destination


def delete_sound(family_folder: str, instrument_folder: str, sound_folder_name: str) -> None:
    folder = SOUND_LIBRARY_ROOT / family_folder / instrument_folder / sound_folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Sound not found: {folder}")

    # Also remove any training-data file a prior correction filed for this
    # sound, so a stray example doesn't outlive it.
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = _read_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            metadata = {}
        correction_file = metadata.get("last_correction_file")
        if correction_file:
            correction_path = Path(correction_file)
            if correction_path.exists():
                # Best-effort: a locked/permission-denied correction file
                # shouldn't block deleting the sound the user actually
                # asked to delete -- that's the primary action here.
                try:
                    correction_path.unlink()
                except OSError:
                    pass

    shutil.rmtree(folder)


def correct_sound_class(
    family_folder: str,
    instrument_folder: str,
    sound_folder_name: str,
    correct_class: str,
    source_audio_path: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """
    Fix a saved sound's CNN classification after the fact, relocating its
    folder to match (a sound's location always mirrors its classification).

    If source_audio_path and data_dir are given, also files that audio as
    a training example under the corrected class, removing any stray
    training file a previous correction left behind first.

    Returns (new saved-sound path, new training-data file path or None).
    """
    source = SOUND_LIBRARY_ROOT / family_folder / instrument_folder / sound_folder_name
    metadata_path = source / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Sound not found: {source}")
    metadata = _read_json(metadata_path)

    # File the training correction while source_audio_path is still valid --
    # it may point inside this sound's own folder, which the move below
    # relocates, invalidating that path.
    training_copy = None
    if source_audio_path is not None and data_dir is not None:
        previous_correction = metadata.get("last_correction_file")
        if previous_correction:
            previous_path = Path(previous_correction)
            if previous_path.exists():
                previous_path.unlink()

        training_copy = add_correction_to_training_data(
            source_audio_path, correct_class, data_dir
        )

    destination_folder = _instrument_folder(correct_class)
    destination = destination_folder / source.name
    if destination != source:
        if destination.exists():
            raise FileExistsError(
                f"A sound named '{source.name}' already exists under "
                f"{display_name(correct_class)}."
            )
        shutil.move(str(source), str(destination))

    metadata_path = destination / "metadata.json"
    metadata["detected_instrument_class"] = correct_class
    metadata["family_folder"] = destination_folder.parent.name
    metadata["instrument_folder"] = destination_folder.name
    if training_copy is not None:
        metadata["last_correction_file"] = str(training_copy)

    _write_json(metadata_path, metadata)
    return destination, training_copy


def add_correction_to_training_data(
    source_audio_path: Path, correct_class: str, data_dir: Path
) -> Path:
    """
    Copy a misclassified sound's original recording into
    data/<family>/<correct_class>/ so the next train_classifier.py run
    learns from the corrected example.
    """
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Missing source audio: {source_audio_path}")

    class_dir = data_dir / family_of(correct_class) / correct_class
    class_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = class_dir / f"correction_{timestamp}_{source_audio_path.stem}.wav"
    shutil.copy2(source_audio_path, destination)
    return destination
