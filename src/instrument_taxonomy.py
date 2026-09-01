"""
instrument_taxonomy.py

Shared hierarchical instrument taxonomy: every instrument the classifier can
recognize belongs to exactly one family. Backed by instrument_taxonomy.json
in this same directory, so new instruments/families can be registered at
runtime (e.g. via the GUI's "+ New Instrument" flow in the CNN-correction
dialogs) without hand-editing this file.

Adding a new instrument:
1. Call add_instrument(name, family) -- or edit instrument_taxonomy.json
   directly and add the family to "families" too if it's new.
2. Drop a data/<family>/<instrument_name>/ folder with its audio (family
   folder name must match the one given in step 1).

No other code changes needed -- train_classifier.py auto-discovers whatever
instrument folders actually have data. A freshly added instrument won't
actually be recognized by the classifier until enough training audio has
been collected for it and train_classifier.py has been re-run.
"""

import json
from pathlib import Path

_TAXONOMY_PATH = Path(__file__).resolve().parent / "instrument_taxonomy.json"


def _load():
    with _TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["families"]), dict(data["instruments"])


def _save():
    _TAXONOMY_PATH.write_text(
        json.dumps(
            {"families": FAMILY_NAMES, "instruments": INSTRUMENT_TO_FAMILY},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


FAMILY_NAMES, INSTRUMENT_TO_FAMILY = _load()
INSTRUMENT_NAMES = list(INSTRUMENT_TO_FAMILY.keys())


def family_of(instrument_name: str) -> str:
    if instrument_name not in INSTRUMENT_TO_FAMILY:
        raise ValueError(
            f"Unknown instrument '{instrument_name}' -- add it via "
            "add_instrument() or src/instrument_taxonomy.json first."
        )
    return INSTRUMENT_TO_FAMILY[instrument_name]


def add_instrument(instrument_name: str, family_name: str, new_family: bool = False) -> None:
    """
    Register a new instrument (and family, if new_family=True), persisted
    to instrument_taxonomy.json. Mutates FAMILY_NAMES/INSTRUMENT_TO_FAMILY/
    INSTRUMENT_NAMES in place, so modules that already imported them see
    the update immediately.

    Only registers the taxonomy entry -- the classifier won't recognize
    the instrument until training audio exists under
    data/<family_name>/<instrument_name>/ and it's retrained.
    """
    if instrument_name in INSTRUMENT_TO_FAMILY:
        raise ValueError(f"'{instrument_name}' is already a known instrument.")

    if new_family:
        if family_name in FAMILY_NAMES:
            raise ValueError(f"Family '{family_name}' already exists.")
        FAMILY_NAMES.append(family_name)
    elif family_name not in FAMILY_NAMES:
        raise ValueError(
            f"Unknown family '{family_name}'. Pass new_family=True to create it."
        )

    INSTRUMENT_TO_FAMILY[instrument_name] = family_name
    INSTRUMENT_NAMES.append(instrument_name)
    _save()
