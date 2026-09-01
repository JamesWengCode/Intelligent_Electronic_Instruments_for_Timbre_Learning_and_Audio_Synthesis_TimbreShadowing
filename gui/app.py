"""
app.py

Playable GUI, 36 notes, middle-C default, ZX octave shift,
timbre memory, EQ, FX, ADSR, volume, and playable keyboard.

Default mode is designed to stay close to the learned/input timbre:
- EQ boost = 0 dB
- Reverb = 0
- Delay = 0
- Attack = 0
- Sustain = 100
- no artificial sustain loop

Run:
python3 gui/app.py

Dependencies:
pip install sounddevice soundfile tkinterdnd2 mido python-rtmidi
"""

import sys
import re
import time
import threading
import subprocess
import shutil
import queue
import json
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter, ImageTk
from pathlib import Path
from tkinter import filedialog, ttk

import safe_dialogs as messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, lfiltic

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import mido
except ImportError:
    mido = None


def find_project_root() -> Path:
    """Find the project folder containing src/run_system.py.

    This works whether this file is stored as gui/app.py or temporarily copied
    elsewhere inside the project.
    """
    script_path = Path(__file__).resolve()
    candidates = [script_path.parent, *script_path.parents]

    for candidate in candidates:
        if (candidate / "src" / "run_system.py").is_file():
            return candidate

    # Preserve the original expected layout as a fallback: project/gui/app.py
    return script_path.parents[1]


PROJECT_ROOT = find_project_root()

# Make project modules importable when gui/app.py is run directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from instrument_taxonomy import (
    FAMILY_NAMES,
    INSTRUMENT_NAMES,
    add_instrument,
    family_of,
)
from mode_selector import get_mode_config

UNCLASSIFIED_VALUES = {"", "not loaded", None}
from sound_library import (
    SOUND_LIBRARY_ROOT,
    add_correction_to_training_data,
    correct_sound_class,
    delete_sound,
    display_name as taxonomy_display_name,
    list_sounds,
    list_taxonomy,
    load_sound,
    rename_sound,
    save_sound,
)

TRAIN_ARGS_STEPS_RE = re.compile(r"Train args:.*'steps':\s*(\d+)")
STEP_LINE_RE = re.compile(r"^Step (\d+)")
RENDERING_COUNT_RE = re.compile(r"Rendering (\d+) notes")
SAVED_WAV_RE = re.compile(r"^Saved .*\.wav$")
PREDICTED_CLASS_LINE_RE = re.compile(r"^predicted_class:.*$", re.MULTILINE)
PREDICTED_FAMILY_LINE_RE = re.compile(r"^predicted_family:.*$", re.MULTILINE)

INPUT_AUDIO = PROJECT_ROOT / "input" / "target.wav"
RUN_SYSTEM = PROJECT_ROOT / "src" / "run_system.py"
TRAIN_CLASSIFIER_SCRIPT = PROJECT_ROOT / "src" / "train_classifier.py"
DATA_DIR = PROJECT_ROOT / "data"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
# Held-out test-set accuracy, written by train_classifier.py at the end
# of its run -- a training/checkpoint diagnostic, not live usage.
TRAINING_METRICS_PATH = PROJECT_ROOT / "models" / "training_metrics.json"
# Real-world "did the user agree with the CNN's prediction" tally --
# see _record_recognition_agreement's docstring. Distinct from the above:
# this is accumulated from actual usage, not a training-time test set.
RECOGNITION_AGREEMENT_PATH = PROJECT_ROOT / "models" / "recognition_agreement.json"

# Rough floor for how many training samples a new instrument needs before
# retraining is likely to recognize it rather than overfit to a handful of
# clips -- based on the smallest class that trains successfully today.
NEW_CLASS_MIN_SAMPLES = 35

# After every 10th correction filed for the same instrument, prompt to
# retrain instead of waiting for the user to remember to click it manually.
RETRAIN_PROMPT_EVERY = 10

LATEST_RUN = PROJECT_ROOT / "outputs" / "latest_run"
SUMMARY_FILE = LATEST_RUN / "run_summary.txt"
RECONSTRUCTED_AUDIO = LATEST_RUN / "timbre_learning" / "output.wav"
TIMBRE_LEARNING_DIR = LATEST_RUN / "timbre_learning"
LEARNED_PARAMS = TIMBRE_LEARNING_DIR / "learned_params.json"
RENDER_SCRIPT = PROJECT_ROOT / "src" / "render_notes.py"
ENHANCE_SCRIPT = PROJECT_ROOT / "src" / "enhance_rendered_notes.py"

# Always the original rendered notes -- a looped "sustain" variant was
# tried before and caused audible stutter; don't reintroduce that.
RENDERED_DIR = LATEST_RUN / "rendered_notes"

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".flac",
    ".ogg",
    ".aif",
    ".aiff",
    ".caf",
}

WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}
BLACK_PITCH_CLASSES = {1, 3, 6, 8, 10}

NOTE_NAMES = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

KEY_TO_OFFSET = {
    "a": 0,  # C
    "w": 1,  # C#
    "s": 2,  # D
    "e": 3,  # D#
    "d": 4,  # E
    "f": 5,  # F
    "t": 6,  # F#
    "g": 7,  # G
    "y": 8,  # G#
    "h": 9,  # A
    "u": 10,  # A#
    "j": 11,  # B
    # Extra upper notes.
    # White keys: K L ; '
    # Black keys: O P [
    "k": 12,  # C next octave
    "o": 13,  # C# next octave
    "l": 14,  # D next octave
    "p": 15,  # D# next octave
    "semicolon": 16,  # E next octave
    "apostrophe": 17,  # F next octave
    "bracketleft": 18,  # F# next octave
}

DEFAULT_CONTROLS = {
    # EQ_FREQ is a normalized 0–100 position and is mapped logarithmically
    # to 80–8000 Hz. 54.9 (not a round number itself) is what actually
    # lands the mapped frequency at a clean "1.00 kHz" display.
    "EQ_FREQ": 54.9,
    "EQ_BOOST": 0,
    "ATTACK": 0,
    "DECAY": 30,
    "SUSTAIN": 100,
    "RELEASE": 25,
    # More headroom. 85 can pop when polyphony + EQ/FX stack.
    "VOLUME": 75,
}

# Anti-click / anti-pop settings.
HEADROOM = 0.82
DECLICK_SECONDS = 0.006
SOFT_LIMIT_DRIVE = 1.35

# A same-note retrigger starts a new overlapping voice instead of cutting
# the old one -- the old voice keeps releasing on its own while the new one
# attacks from zero. Caps how many voices one note can pile up under fast
# repeated retriggering.
MAX_VOICES_PER_NOTE = 4

# Per-note one-pole output smoothing: y[n] = SMOOTH_ALPHA*x[n] + (1-SMOOTH_ALPHA)*y[n-1].
SMOOTH_ALPHA = 0.92

# Pitch bend / mod wheel. Mod wheel drives vibrato depth (a small
# periodic pitch wobble) at a fixed, typical vibrato rate -- CC1 controls
# how deep the wobble is, not how fast.
PITCH_BEND_RANGE_SEMITONES = 1.0
VIBRATO_RATE_HZ = 5.5
VIBRATO_DEPTH_SEMITONES = 0.4

# Balances loudness against per-block polyphony compensation without
# exposing the additive synthesis's artifacts as much as full gain (1.0)
# would.
MASTER_VOICE_GAIN = 0.58
LIMITER_THRESHOLD = 0.86
LIMITER_ATTACK_SECONDS = 0.002
LIMITER_RELEASE_SECONDS = 0.120
POLYPHONY_COMPENSATION_EXPONENT = 0.50
POLYPHONY_GAIN_SMOOTH_SECONDS = 0.025

INPUT_PANEL_CONTENT_WIDTH = 130
OUTPUT_PANEL_CONTENT_WIDTH = 130


def midi_to_note_name(midi: int) -> str:
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def parse_midi_from_filename(path: Path) -> int | None:
    parts = path.stem.split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return None


def normalize_key(keysym: str) -> str:
    key = keysym.lower()

    # Tk may report punctuation keys in different forms.
    if key in {";", "semicolon"}:
        return "semicolon"

    if key in {"'", "apostrophe", "quoteright", "quotedbl"}:
        return "apostrophe"

    if key in {"[", "bracketleft"}:
        return "bracketleft"

    return key


def normalized_to_log_frequency(value: float) -> float:
    """Map a 0–100 GUI position logarithmically to 80–8000 Hz."""
    value = max(0.0, min(100.0, float(value)))
    minimum = 80.0
    maximum = 8000.0
    return minimum * ((maximum / minimum) ** (value / 100.0))


def map_attack(value: float) -> float:
    """Map 0–100 to 0–1.5 seconds with finer control near zero."""
    normalized = max(0.0, min(100.0, float(value))) / 100.0
    return 1.5 * (normalized**2)


def map_decay(value: float) -> float:
    """Map 0–100 to 0.05–2.0 seconds."""
    normalized = max(0.0, min(100.0, float(value))) / 100.0
    return 0.05 + 1.95 * (normalized**2)


def map_release(value: float) -> float:
    """Map 0–100 to 0.05–3.0 seconds."""
    normalized = max(0.0, min(100.0, float(value))) / 100.0
    return 0.05 + 2.95 * (normalized**2)


def volume_percent_to_gain(percent: float) -> float:
    """Map the 0-100 VOLUME fader to gain using a dB taper (0% -> silence,
    100% -> unity), since loudness perception is roughly logarithmic and a
    linear-amplitude fader would pack all the audible change into the
    bottom ~20% of its travel."""
    normalized = max(0.0, min(100.0, float(percent))) / 100.0
    if normalized <= 0.0:
        return 0.0
    # -30dB range gives a gentler slope per % of drag than a wider range
    # would; 0% still lands on true silence via the early return above.
    min_db = -30.0
    db = min_db * (1.0 - normalized)
    return 10.0 ** (db / 20.0)


def log_frequency_to_normalized(frequency: float) -> float:
    minimum = 80.0
    maximum = 8000.0
    frequency = max(minimum, min(maximum, float(frequency)))
    return 100.0 * np.log(frequency / minimum) / np.log(maximum / minimum)


def attack_seconds_to_control(seconds: float) -> float:
    seconds = max(0.0, min(1.5, float(seconds)))
    return 100.0 * np.sqrt(seconds / 1.5) if seconds > 0 else 0.0


def decay_seconds_to_control(seconds: float) -> float:
    seconds = max(0.05, min(2.0, float(seconds)))
    return 100.0 * np.sqrt((seconds - 0.05) / 1.95)


def release_seconds_to_control(seconds: float) -> float:
    seconds = max(0.05, min(3.0, float(seconds)))
    return 100.0 * np.sqrt((seconds - 0.05) / 2.95)


def control_scale_marks(name: str):
    """Return true feature-specific major scale marks."""
    if name == "EQ_FREQ":
        frequencies = [8000, 4000, 2000, 1000, 500, 200, 80]
        return [
            (
                log_frequency_to_normalized(freq),
                f"{freq // 1000}k" if freq >= 1000 else str(freq),
            )
            for freq in frequencies
        ]

    if name == "EQ_BOOST":
        return [
            (12, "+12"),
            (6, "+6"),
            (0, "0"),
            (-6, "-6"),
            (-12, "-12"),
        ]

    if name in {"SUSTAIN", "VOLUME"}:
        return [
            (100, "100%"),
            (75, "75%"),
            (50, "50%"),
            (25, "25%"),
            (0, "0%"),
        ]

    if name == "ATTACK":
        return [
            (attack_seconds_to_control(1.5), "1.5 s"),
            (attack_seconds_to_control(1.0), "1.0"),
            (attack_seconds_to_control(0.5), "0.5"),
            (attack_seconds_to_control(0.2), "0.2"),
            (attack_seconds_to_control(0.05), "0.05"),
            (0, "0"),
        ]

    if name == "DECAY":
        return [
            (decay_seconds_to_control(2.0), "2.0 s"),
            (decay_seconds_to_control(1.5), "1.5"),
            (decay_seconds_to_control(1.0), "1.0"),
            (decay_seconds_to_control(0.5), "0.5"),
            (decay_seconds_to_control(0.1), "0.1"),
            (0, "0.05"),
        ]

    if name == "RELEASE":
        return [
            (release_seconds_to_control(3.0), "3.0 s"),
            (release_seconds_to_control(2.0), "2.0"),
            (release_seconds_to_control(1.0), "1.0"),
            (release_seconds_to_control(0.5), "0.5"),
            (release_seconds_to_control(0.1), "0.1"),
            (0, "0.05"),
        ]

    return []


def create_metal_handle_image(width, height, pad=8, radius=4):
    """Shared metallic gradient handle image (drop shadow, embossed center
    ribs) used by both MixerFader and Wheel, so every draggable control in
    the app reads as the same physical material."""
    base = Image.new("RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sdw = ImageDraw.Draw(shadow)
    sdw.rounded_rectangle(
        (pad + 3, pad + 4, pad + width + 3, pad + height + 4),
        radius=radius,
        fill=(0, 0, 0, 165),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(3))
    base.alpha_composite(shadow)

    body = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=radius, fill=255
    )
    grad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    px = grad.load()
    bands = [
        (0.0, (232, 233, 234)),
        (0.16, (173, 176, 179)),
        (0.42, (94, 97, 101)),
        (0.56, (218, 220, 221)),
        (0.76, (137, 140, 143)),
        (1.0, (68, 70, 73)),
    ]
    for y in range(height):
        t = y / max(1, height - 1)
        color = bands[-1][1]
        for i in range(len(bands) - 1):
            t0, c0 = bands[i]
            t1, c1 = bands[i + 1]
            if t0 <= t <= t1:
                q = (t - t0) / max(1e-9, t1 - t0)
                color = tuple(int(c0[j] * (1 - q) + c1[j] * q) for j in range(3))
                break
        for x in range(width):
            edge = abs(x / max(1, width - 1) - 0.5) * 2
            d = int(edge * 18)
            px[x, y] = (
                max(0, color[0] - d),
                max(0, color[1] - d),
                max(0, color[2] - d),
                255,
            )
    body.paste(grad, (0, 0), mask)
    draw = ImageDraw.Draw(body)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=radius,
        outline=(35, 36, 38, 255),
        width=2,
    )
    # subtle embossed center ribs like the reference
    cy = height // 2
    draw.line((3, cy - 5, width - 4, cy - 5), fill=(235, 235, 235, 145), width=1)
    draw.line((3, cy, width - 4, cy), fill=(250, 250, 250, 255), width=2)
    draw.line((3, cy + 3, width - 4, cy + 3), fill=(55, 57, 59, 220), width=1)
    base.alpha_composite(body, (pad, pad))
    return ImageTk.PhotoImage(base)


class NumberStepper(tk.Frame):
    """Small dark numeric field with up/down arrow buttons, used for the
    pitch bend up/down range controls. A plain tk.Spinbox's increment/
    decrement buttons render inconsistently on macOS Aqua (like tk.Button,
    they partly ignore color customization -- see MixerFader's comments
    elsewhere in this file for the same issue), so this is hand-drawn
    instead, matching the rest of this app's custom-widget approach."""

    def __init__(self, parent, initial, minimum, maximum, command=None, width=3):
        super().__init__(parent, bg="#171817")
        self.minimum = minimum
        self.maximum = maximum
        self.command = command
        self.value = int(initial)

        self.var = tk.StringVar(value=str(self.value))
        entry = tk.Entry(
            self,
            textvariable=self.var,
            width=width,
            justify="center",
            bg="#0c0c0c",
            fg="#f2f2f2",
            insertbackground="#f2f2f2",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#3a3b3c",
            highlightcolor="#676767",
            font=("Segoe UI", 12, "bold"),
        )
        entry.pack(side="left", ipady=3)
        entry.bind("<Return>", self._on_entry_commit)
        entry.bind("<FocusOut>", self._on_entry_commit)

        stepper = tk.Frame(self, bg="#171817")
        stepper.pack(side="left", padx=(2, 0))
        self.up_canvas = tk.Canvas(
            stepper, width=14, height=11, bg="#171817",
            highlightthickness=0, bd=0, cursor="hand2",
        )
        self.up_triangle = self.up_canvas.create_polygon(
            2, 9, 12, 9, 7, 2, fill="#aeb2b5", outline=""
        )
        self.up_canvas.pack()
        self.up_canvas.bind("<Button-1>", lambda _e: self._step(1))
        self.down_canvas = tk.Canvas(
            stepper, width=14, height=11, bg="#171817",
            highlightthickness=0, bd=0, cursor="hand2",
        )
        self.down_triangle = self.down_canvas.create_polygon(
            2, 2, 12, 2, 7, 9, fill="#aeb2b5", outline=""
        )
        self.down_canvas.pack()
        self.down_canvas.bind("<Button-1>", lambda _e: self._step(-1))

        self._update_limit_look()

    def _step(self, delta):
        self.set(self.value + delta)
        self.notify()

    def _on_entry_commit(self, _event):
        try:
            value = int(float(self.var.get()))
        except ValueError:
            value = self.value
        self.set(value)
        self.notify()

    def set(self, value):
        value = max(self.minimum, min(self.maximum, int(value)))
        self.value = value
        self.var.set(str(value))
        self._update_limit_look()

    def _update_limit_look(self):
        """Dims whichever arrow is already at its limit (and can't
        actually do anything if clicked) instead of leaving it looking
        just as clickable as the other one."""
        at_max = self.value >= self.maximum
        self.up_canvas.itemconfig(
            self.up_triangle, fill="#45464a" if at_max else "#aeb2b5"
        )
        self.up_canvas.config(cursor="arrow" if at_max else "hand2")

        at_min = self.value <= self.minimum
        self.down_canvas.itemconfig(
            self.down_triangle, fill="#45464a" if at_min else "#aeb2b5"
        )
        self.down_canvas.config(cursor="arrow" if at_min else "hand2")

    def get(self):
        return self.value

    def notify(self):
        if self.command:
            self.command(self.value)


class MixerFader(tk.Canvas):
    """Tall dark studio fader matching the reference UI."""

    def __init__(
        self,
        parent,
        label,
        initial=50,
        min_value=0,
        max_value=100,
        resolution=1,
        command=None,
        value_formatter=None,
        scale_marks=None,
        width=116,
        height=520,
        track_top=102,
        bottom_margin=82,
        label_y=60,
        label_fg="#f2f2f2",
        show_value_text=True,
        spring_back_to=None,
        on_label_click=None,
    ):
        self.bg_color = "#202120"
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=self.bg_color,
            highlightthickness=0,
            bd=0,
        )
        self.label = label
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.resolution = float(resolution)
        self.command = command
        self.value_formatter = value_formatter
        self.scale_marks = scale_marks or []
        self.width = width
        self.height = height
        self.track_x = int(width * 0.40)
        self.track_top = track_top
        self.track_bottom = height - bottom_margin
        self.label_y = label_y
        self.label_fg = label_fg
        self.show_value_text = show_value_text
        self.handle_width = 30
        self.handle_height = 46
        self.dragging = False
        self.drag_offset_y = 0
        self.spring_back_to = spring_back_to
        self.default_value = self.clamp(float(initial))
        self.value = self.default_value
        self.draw_static()
        self.draw_handle()
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Double-Button-1>", self.on_double_click)
        self.bind("<MouseWheel>", self.on_wheel)
        if on_label_click is not None:
            self.tag_bind("label", "<Button-1>", lambda _e: on_label_click())
            self.tag_bind("label", "<Enter>", lambda _e: self.config(cursor="hand2"))
            self.tag_bind("label", "<Leave>", lambda _e: self.config(cursor=""))

    def clamp(self, value):
        value = max(self.min_value, min(self.max_value, float(value)))
        if self.resolution > 0:
            steps = round((value - self.min_value) / self.resolution)
            value = self.min_value + steps * self.resolution
        return max(self.min_value, min(self.max_value, value))

    def value_to_y_for(self, value):
        span = self.max_value - self.min_value
        if span <= 0:
            return self.track_bottom
        ratio = (float(value) - self.min_value) / span
        return self.track_bottom - ratio * (self.track_bottom - self.track_top)

    def value_to_y(self):
        return self.value_to_y_for(self.value)

    def y_to_value(self, y):
        y = max(self.track_top, min(self.track_bottom, y))
        ratio = (self.track_bottom - y) / (self.track_bottom - self.track_top)
        return self.clamp(self.min_value + ratio * (self.max_value - self.min_value))

    def formatted_value(self):
        return (
            self.value_formatter(self.value)
            if self.value_formatter
            else f"{self.value:g}"
        )

    def draw_static(self):
        self.label_item = self.create_text(
            self.width // 2,
            self.label_y,
            text=self.label,
            fill=self.label_fg,
            font=("Segoe UI", 12, "bold"),
            tags=("label",),
        )
        self.create_rectangle(
            self.track_x - 7,
            self.track_top,
            self.track_x + 7,
            self.track_bottom,
            fill="#050606",
            outline="#090909",
            width=2,
        )
        self.create_line(
            self.track_x - 4,
            self.track_top + 2,
            self.track_x - 4,
            self.track_bottom - 2,
            fill="#000000",
            width=2,
        )
        self.create_line(
            self.track_x + 4,
            self.track_top + 2,
            self.track_x + 4,
            self.track_bottom - 2,
            fill="#333536",
            width=1,
        )

        scale_x = self.track_x + 18

        # Draw only parameter-specific ticks. The previous generic 0–100
        # ticks were visually misleading for logarithmic frequency and
        # nonlinear ADSR mappings.
        sorted_marks = sorted(
            self.scale_marks,
            key=lambda item: float(item[0]),
        )

        # Two small intermediate ticks between adjacent labeled values.
        for index in range(len(sorted_marks) - 1):
            value_a = float(sorted_marks[index][0])
            value_b = float(sorted_marks[index + 1][0])

            for fraction in (1 / 3, 2 / 3):
                minor_value = value_a + (value_b - value_a) * fraction
                y = self.value_to_y_for(minor_value)
                self.create_line(
                    scale_x,
                    y,
                    scale_x + 5,
                    y,
                    fill="#8e9092",
                    width=1,
                )

        # Labeled major ticks use the actual mapping for each feature.
        for mark_value, mark_label in self.scale_marks:
            y = self.value_to_y_for(mark_value)
            self.create_line(
                scale_x - 1,
                y,
                scale_x + 11,
                y,
                fill="#eeeeee",
                width=1,
            )
            self.create_text(
                scale_x + 16,
                y,
                text=mark_label,
                anchor="w",
                fill="#f1f1f1",
                font=("Segoe UI", 9),
            )

        if self.show_value_text:
            self.create_text(
                self.width // 2,
                self.height - 26,
                text=self.formatted_value(),
                fill="#f8d36a",
                font=("Segoe UI", 11, "bold"),
                tags=("value_text",),
            )

    def draw_handle(self):
        self.handle_image = create_metal_handle_image(
            self.handle_width, self.handle_height
        )
        self.handle_item = self.create_image(
            self.track_x, self.value_to_y(), image=self.handle_image, tags=("handle",)
        )
        value_text_items = self.find_withtag("value_text")
        self.value_text = value_text_items[0] if value_text_items else None

    def update_handle(self):
        self.coords(self.handle_item, self.track_x, self.value_to_y())
        if self.value_text is not None:
            self.itemconfig(self.value_text, text=self.formatted_value())

    def get(self):
        return self.value

    def set(self, value):
        self.value = self.clamp(value)
        self.update_handle()

    def set_label(self, text):
        self.label = text
        self.itemconfig(self.label_item, text=text)

    def on_press(self, event):
        # Only grabbing the handle itself starts a drag -- a stray click
        # elsewhere on the track shouldn't jump the value to that position.
        current_y = self.value_to_y()
        within_y = abs(event.y - current_y) <= self.handle_height / 2
        within_x = abs(event.x - self.track_x) <= self.handle_width / 2
        if within_y and within_x:
            self.drag_offset_y = event.y - current_y
            self.dragging = True

    def on_drag(self, event):
        if self.dragging:
            self.value = self.y_to_value(event.y - self.drag_offset_y)
            self.update_handle()
            self.notify()

    def on_release(self, _event):
        if not self.dragging:
            return
        self.dragging = False
        if self.spring_back_to is not None:
            self.value = self.clamp(self.spring_back_to)
            self.update_handle()
            self.notify()

    def on_double_click(self, _event):
        self.dragging = False
        self.value = self.default_value
        self.update_handle()
        self.notify()

    def on_wheel(self, event):
        self.value = self.clamp(
            self.value + (1 if event.delta > 0 else -1) * self.resolution
        )
        self.update_handle()
        self.notify()

    def notify(self):
        if self.command:
            self.command(self.value)


class AudioEngine:
    def __init__(self, sample_rate=16000, blocksize=512):
        if sd is None:
            raise ImportError(
                "sounddevice is not installed. Run: pip install sounddevice"
            )

        self.sample_rate = sample_rate
        self.blocksize = blocksize

        # Default mode: close to original learned/input sound.
        self.volume = 0.85
        self.eq_freq_hz = 1000.0
        self.eq_gain_db = 0.0
        self.reverb_mix = 0.0
        self.delay_mix = 0.0

        self.attack_seconds = 0.0
        self.decay_seconds = 0.60
        self.sustain_level = 1.0
        self.release_seconds = 0.60

        # Pitch bend (semitone offset, -PITCH_BEND_RANGE..+PITCH_BEND_RANGE)
        # and mod wheel (0..1, drives vibrato depth). Both smoothed like the
        # other controls below so rapid wheel motion doesn't zipper.
        self.pitch_bend_semitones = 0.0
        self.mod_wheel = 0.0
        self.vibrato_phase = 0.0

        # Parameter targets are smoothed in the audio callback to reduce
        # clicks/pops when a fader moves abruptly.
        self.target_volume = self.volume
        self.target_eq_freq_hz = self.eq_freq_hz
        self.target_eq_gain_db = self.eq_gain_db
        self.target_reverb_mix = self.reverb_mix
        self.target_delay_mix = self.delay_mix
        self.target_attack_seconds = self.attack_seconds
        self.target_decay_seconds = self.decay_seconds
        self.target_sustain_level = self.sustain_level
        self.target_release_seconds = self.release_seconds
        self.target_pitch_bend_semitones = self.pitch_bend_semitones
        self.target_mod_wheel = self.mod_wheel
        self.control_smoothing_seconds = 0.060
        self.declick_samples = max(1, int(DECLICK_SECONDS * self.sample_rate))

        self.note_buffers = {}
        self.active_notes = {}
        self.lock = threading.Lock()

        # Transparent master limiter state.
        self.master_gain = 1.0

        # Meter points:
        # input_level_peak = after voices/FX, before the Volume fader
        # output_level_peak = after Volume and limiter
        self.input_level_peak = 0.0
        self.output_level_peak = 0.0
        self.clip_detected = False

        # Transparent polyphony compensation.
        # 1 voice = 1.0, 2 voices ≈ 0.707, 3 voices ≈ 0.577.
        self.polyphony_gain = 1.0

        self.delay_buffer = np.zeros(
            max(1, int(self.sample_rate * 0.35)), dtype=np.float32
        )
        self.delay_index = 0
        self.reverb_buffer = np.zeros(
            max(1, int(self.sample_rate * 0.085)), dtype=np.float32
        )
        self.reverb_index = 0

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.callback,
        )
        self.stream.start()

    def reset_fx_buffers(self):
        self.delay_buffer = np.zeros(
            max(1, int(self.sample_rate * 0.35)), dtype=np.float32
        )
        self.delay_index = 0
        self.reverb_buffer = np.zeros(
            max(1, int(self.sample_rate * 0.085)), dtype=np.float32
        )
        self.reverb_index = 0

    def set_controls(
        self,
        volume=None,
        eq_freq_hz=None,
        eq_gain_db=None,
        reverb_mix=None,
        delay_mix=None,
        attack_seconds=None,
        decay_seconds=None,
        sustain_level=None,
        release_seconds=None,
        pitch_bend_semitones=None,
        mod_wheel=None,
    ):
        with self.lock:
            if volume is not None:
                self.target_volume = float(volume)
            if eq_freq_hz is not None:
                self.target_eq_freq_hz = float(eq_freq_hz)
            if eq_gain_db is not None:
                self.target_eq_gain_db = float(eq_gain_db)
            if reverb_mix is not None:
                self.target_reverb_mix = float(reverb_mix)
            if delay_mix is not None:
                self.target_delay_mix = float(delay_mix)
            if attack_seconds is not None:
                self.target_attack_seconds = float(attack_seconds)
            if decay_seconds is not None:
                self.target_decay_seconds = float(decay_seconds)
            if sustain_level is not None:
                self.target_sustain_level = float(sustain_level)
            if release_seconds is not None:
                self.target_release_seconds = float(release_seconds)
            if pitch_bend_semitones is not None:
                self.target_pitch_bend_semitones = float(pitch_bend_semitones)
            if mod_wheel is not None:
                self.target_mod_wheel = float(mod_wheel)

    def smooth_controls(self, frames):
        smoothing_samples = max(
            1.0,
            self.control_smoothing_seconds * self.sample_rate,
        )
        alpha = 1.0 - np.exp(-frames / smoothing_samples)

        self.volume += (self.target_volume - self.volume) * alpha
        self.eq_freq_hz += (self.target_eq_freq_hz - self.eq_freq_hz) * alpha
        self.eq_gain_db += (self.target_eq_gain_db - self.eq_gain_db) * alpha
        self.reverb_mix += (self.target_reverb_mix - self.reverb_mix) * alpha
        self.delay_mix += (self.target_delay_mix - self.delay_mix) * alpha
        self.attack_seconds += (
            self.target_attack_seconds - self.attack_seconds
        ) * alpha
        self.decay_seconds += (self.target_decay_seconds - self.decay_seconds) * alpha
        self.sustain_level += (self.target_sustain_level - self.sustain_level) * alpha
        self.release_seconds += (
            self.target_release_seconds - self.release_seconds
        ) * alpha
        self.pitch_bend_semitones += (
            self.target_pitch_bend_semitones - self.pitch_bend_semitones
        ) * alpha
        self.mod_wheel += (self.target_mod_wheel - self.mod_wheel) * alpha

    def get_meter_levels(self):
        with self.lock:
            return {
                "input_level_peak": float(self.input_level_peak),
                "output_level_peak": float(self.output_level_peak),
                "clip_detected": bool(self.clip_detected),
            }

    def close(self):
        with self.lock:
            self.active_notes.clear()
        self.stream.stop()
        self.stream.close()

    def load_notes(self, wav_by_midi):
        buffers = {}
        target_sr = None

        for midi, path in wav_by_midi.items():
            y, sr = sf.read(path)

            if y.ndim > 1:
                y = y.mean(axis=1)

            y = y.astype(np.float32)
            y = y - np.mean(y)

            peak = np.max(np.abs(y)) + 1e-8
            if peak > 0.98:
                y = y / peak * 0.98

            if target_sr is None:
                target_sr = sr

            if sr != target_sr:
                raise RuntimeError(
                    f"Different sample rates are not supported yet: {path}"
                )

            buffers[midi] = y

        if target_sr is not None and target_sr != self.sample_rate:
            self.stream.stop()
            self.stream.close()
            self.sample_rate = target_sr
            self.declick_samples = max(1, int(DECLICK_SECONDS * self.sample_rate))
            self.reset_fx_buffers()
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                callback=self.callback,
            )
            self.stream.start()

        with self.lock:
            self.note_buffers = buffers
            self.active_notes.clear()
            self.reset_fx_buffers()

    def peaking_eq_coefficients(self):
        # RBJ-style peaking EQ.
        freq = min(max(self.eq_freq_hz, 40.0), self.sample_rate * 0.45)
        gain_db = self.eq_gain_db

        # When boost is neutral, bypass exactly enough.
        if abs(gain_db) < 0.05:
            return None

        q = 1.0
        a = 10 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2.0 * q)

        b0 = 1.0 + alpha * a
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / a

        return (
            b0 / a0,
            b1 / a0,
            b2 / a0,
            a1 / a0,
            a2 / a0,
        )

    def note_on(self, midi, velocity=127):
        with self.lock:
            if midi not in self.note_buffers:
                return False

            velocity = max(1, min(127, int(velocity)))
            velocity_gain = (velocity / 127.0) ** 0.60

            voices = self.active_notes.setdefault(midi, [])

            # A retrigger releases whichever voice was already held instead
            # of cutting it -- it decays on its own while the new voice
            # attacks from zero and is summed with it in the mixer.
            for voice in voices:
                if voice["stage"] != "release":
                    voice["stage"] = "release"

            # Cap overlapping voices per note under very fast repeated
            # retriggering -- drop the oldest (already releasing) one.
            if len(voices) >= MAX_VOICES_PER_NOTE:
                voices.pop(0)

            voices.append(
                {
                    # Float, not int -- pitch bend/vibrato read through y at
                    # a variable (non-1.0) rate, so the position advances by
                    # fractional amounts each block.
                    "pos": 0.0,
                    "stage": "attack",
                    # Always use a short internal attack for de-clicking,
                    # even when the visible ATTACK control is zero.
                    "env": 0.0,
                    "velocity_gain": velocity_gain,
                    "eq_zi": np.zeros(2, dtype=np.float64),
                    "smooth_zi": lfiltic(
                        [SMOOTH_ALPHA], [1.0, -(1.0 - SMOOTH_ALPHA)], y=[0.0]
                    ),
                    "prev_out": 0.0,
                }
            )

            return True

    def note_off(self, midi):
        with self.lock:
            for voice in self.active_notes.get(midi, []):
                if voice["stage"] != "release":
                    voice["stage"] = "release"

    def stop_all(self):
        with self.lock:
            self.active_notes.clear()

    def generate_envelope_block(self, state, n):
        """
        Vectorized replacement for the old per-sample ADSR update.

        Generates up to n envelope samples using closed-form segments
        (linear ramp for attack, geometric decay for decay/release,
        constant for sustain), looping only over stage *transitions*
        (at most a handful per call) instead of over every sample.

        Returns (envelope: np.ndarray of length <= n, remove_after_release: bool).
        """
        chunks = []
        filled = 0
        stage = state["stage"]
        env = state["env"]
        remove_after_release = False

        while filled < n:
            remaining = n - filled

            if stage == "attack":
                # A minimum 6 ms attack avoids clicks when ATTACK is set to zero.
                effective_attack = max(self.attack_seconds, DECLICK_SECONDS)
                inc = 1.0 / max(1.0, effective_attack * self.sample_rate)
                ramp = env + inc * np.arange(1, remaining + 1)
                cross = np.searchsorted(ramp, 1.0, side="left")

                if cross < remaining:
                    seg_len = int(cross) + 1
                    seg = ramp[:seg_len].copy()
                    seg[-1] = 1.0
                    chunks.append(seg)
                    filled += seg_len
                    env = 1.0
                    stage = "decay"
                else:
                    chunks.append(ramp)
                    filled += remaining
                    env = float(ramp[-1])

            elif stage == "decay":
                target = self.sustain_level
                coeff = np.exp(-1.0 / max(1.0, self.decay_seconds * self.sample_rate))
                powers = coeff ** np.arange(1, remaining + 1)
                seg = target + (env - target) * powers
                close = np.nonzero(np.abs(seg - target) < 0.003)[0]

                if close.size > 0:
                    seg_len = int(close[0]) + 1
                    seg = seg[:seg_len].copy()
                    seg[-1] = target
                    chunks.append(seg)
                    filled += seg_len
                    env = target
                    stage = "sustain"
                else:
                    chunks.append(seg)
                    filled += remaining
                    env = float(seg[-1])

            elif stage == "sustain":
                seg = np.full(remaining, self.sustain_level)
                chunks.append(seg)
                filled += remaining
                env = self.sustain_level

            elif stage == "release":
                coeff = np.exp(-1.0 / max(1.0, self.release_seconds * self.sample_rate))
                powers = coeff ** np.arange(1, remaining + 1)
                seg = env * powers
                below = np.nonzero(seg < 0.002)[0]

                if below.size > 0:
                    seg_len = int(below[0]) + 1
                    chunks.append(seg[:seg_len])
                    filled += seg_len
                    env = float(seg[seg_len - 1])
                    remove_after_release = True
                    break
                else:
                    chunks.append(seg)
                    filled += remaining
                    env = float(seg[-1])

            else:
                break

        state["stage"] = stage
        state["env"] = env

        if chunks:
            envelope = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        else:
            envelope = np.zeros(0)

        return envelope, remove_after_release

    def run_feedback_buffer(self, dry, buffer, index, feedback):
        """
        Vectorized circular-buffer read/write for a simple feedback delay line.

        Relies on len(buffer) >= len(dry) (true here: delay/reverb buffers are
        hundreds of ms long, always far bigger than one audio block), so reads
        for this block never alias writes made earlier in the same block.
        """
        n = len(dry)
        blen = len(buffer)
        idx = (index + np.arange(n)) % blen
        read = buffer[idx].astype(np.float64)
        buffer[idx] = dry + read * feedback
        new_index = int((index + n) % blen)
        return read, new_index

    def apply_fx(self, dry):
        # Bypass when both are zero, so default is exactly dry/original.
        if self.delay_mix <= 0.001 and self.reverb_mix <= 0.001:
            return dry

        dry64 = dry.astype(np.float64, copy=False)

        # Delay: quieter, cleaner echo.
        delay_read, self.delay_index = self.run_feedback_buffer(
            dry64, self.delay_buffer, self.delay_index, 0.22
        )

        # Reverb: very simple short ambience. Low feedback avoids metallic ringing.
        reverb_read, self.reverb_index = self.run_feedback_buffer(
            dry64, self.reverb_buffer, self.reverb_index, 0.38
        )

        wet_delay = delay_read * self.delay_mix
        wet_reverb = reverb_read * self.reverb_mix * 0.45

        return (dry64 + wet_delay + wet_reverb).astype(dry.dtype)

    def get_polyphony_target_gain(self, voice_count: int) -> float:
        voice_count = max(1, int(voice_count))
        return 1.0 / (voice_count**POLYPHONY_COMPENSATION_EXPONENT)

    def apply_transparent_limiter(self, audio: np.ndarray) -> np.ndarray:
        """
        Transparent peak limiter with gain smoothing.

        Unlike tanh saturation or per-block normalization, this only reduces
        gain when the summed signal would exceed LIMITER_THRESHOLD.
        """
        if len(audio) == 0:
            return audio.astype(np.float32)

        peak = float(np.max(np.abs(audio)) + 1e-8)

        if peak > LIMITER_THRESHOLD:
            target_gain = LIMITER_THRESHOLD / peak
            time_constant = LIMITER_ATTACK_SECONDS
        else:
            target_gain = 1.0
            time_constant = LIMITER_RELEASE_SECONDS

        smoothing_samples = max(1.0, time_constant * self.sample_rate)
        alpha = 1.0 - np.exp(-len(audio) / smoothing_samples)

        self.master_gain += (target_gain - self.master_gain) * alpha
        self.master_gain = max(0.0, min(1.0, self.master_gain))

        return (audio * self.master_gain).astype(np.float32)

    def render_note_block(self, state, y, eq_coeffs, frames, pitch_ratio=1.0):
        """
        Vectorized rendering of up to `frames` samples of one voice (EQ,
        ADSR envelope, edge declick fades, output smoothing), so the audio
        callback keeps up with many overlapping notes in real time.

        pitch_ratio != 1.0 reads through `y` at a different rate via linear
        interpolation, without affecting the ADSR envelope's own timing.

        Returns (note_out: np.ndarray[frames] or None, should_remove: bool).
        """
        pos = state["pos"]
        pitch_ratio = max(0.1, float(pitch_ratio))

        # How many OUTPUT samples fit before the (fractional) read position
        # would run past the last usable index in y (idx0+1 must stay in
        # bounds for interpolation).
        remaining_input = (len(y) - 1) - pos
        if remaining_input <= 0:
            return None, True
        n = min(frames, int(remaining_input / pitch_ratio))

        if n <= 0:
            return None, True

        read_positions = pos + np.arange(n) * pitch_ratio
        idx0 = read_positions.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, len(y) - 1)
        frac = read_positions - idx0
        raw = (y[idx0] * (1.0 - frac) + y[idx1] * frac).astype(np.float64)

        if eq_coeffs is not None:
            b0, b1, b2, a1, a2 = eq_coeffs
            filtered, state["eq_zi"] = lfilter(
                [b0, b1, b2], [1.0, a1, a2], raw, zi=state["eq_zi"]
            )
        else:
            filtered = raw

        envelope, remove_after_release = self.generate_envelope_block(state, n)
        filled = len(envelope)

        if filled == 0:
            state["pos"] = pos
            return None, True

        velocity_gain = state.get("velocity_gain", 1.0)
        processed = filtered[:filled] * envelope * velocity_gain

        abs_pos = read_positions[:filled]

        # Tiny fade-in even when ATTACK is 0.
        # Prevents click if the rendered waveform starts away from zero.
        fade_in_mask = abs_pos < self.declick_samples
        if np.any(fade_in_mask):
            processed[fade_in_mask] *= abs_pos[fade_in_mask] / max(
                1, self.declick_samples
            )

        # Tiny fade-out near the sample end.
        buffer_remaining = (len(y) - 1) - abs_pos
        fade_out_mask = buffer_remaining < self.declick_samples
        if np.any(fade_out_mask):
            processed[fade_out_mask] *= np.maximum(
                0.0, buffer_remaining[fade_out_mask] / max(1, self.declick_samples)
            )

        # Per-note smoothing reduces sudden EQ/level jumps.
        smoothed, state["smooth_zi"] = lfilter(
            [SMOOTH_ALPHA],
            [1.0, -(1.0 - SMOOTH_ALPHA)],
            processed,
            zi=state["smooth_zi"],
        )
        state["prev_out"] = float(smoothed[-1])
        state["pos"] = pos + filled * pitch_ratio

        should_remove = remove_after_release or (state["pos"] >= len(y) - 1)

        note_out = np.zeros(frames, dtype=np.float64)
        note_out[:filled] = smoothed
        return note_out, should_remove

    def callback(self, outdata, frames, time, status):
        if status:
            print(f"[AudioEngine] callback status: {status}", flush=True)

        out = np.zeros(frames, dtype=np.float64)

        with self.lock:
            self.smooth_controls(frames)
            eq_coeffs = self.peaking_eq_coefficients()
            remove_notes = []

            # Pitch bend and mod-wheel vibrato apply channel-wide, computed
            # once per block. Vibrato LFO phase free-runs across blocks/
            # notes, independent of note-on/off.
            self.vibrato_phase = (
                self.vibrato_phase + 2.0 * np.pi * VIBRATO_RATE_HZ * frames / self.sample_rate
            ) % (2.0 * np.pi)
            vibrato_semitones = (
                self.mod_wheel * VIBRATO_DEPTH_SEMITONES * np.sin(self.vibrato_phase)
            )
            pitch_ratio = 2.0 ** (
                (self.pitch_bend_semitones + vibrato_semitones) / 12.0
            )

            for midi, voices in list(self.active_notes.items()):
                y = self.note_buffers.get(midi)

                if y is None or len(y) == 0:
                    remove_notes.append(midi)
                    continue

                still_active = []
                for voice in voices:
                    note_out, should_remove = self.render_note_block(
                        voice, y, eq_coeffs, frames, pitch_ratio
                    )

                    if note_out is not None:
                        out += note_out

                    if not should_remove:
                        still_active.append(voice)

                if still_active:
                    self.active_notes[midi] = still_active
                else:
                    remove_notes.append(midi)

            for midi in set(remove_notes):
                self.active_notes.pop(midi, None)

        # Polyphony compensation is based on distinct notes sounding, not
        # total overlapping voices -- a fast retrigger's brief extra
        # releasing voice is still perceptually "one note" and shouldn't
        # duck the mix like an actual chord would.
        voice_count = max(1, len(self.active_notes))
        target_polyphony_gain = self.get_polyphony_target_gain(voice_count)
        smoothing_samples = max(1.0, POLYPHONY_GAIN_SMOOTH_SECONDS * self.sample_rate)
        poly_alpha = 1.0 - np.exp(-frames / smoothing_samples)
        self.polyphony_gain += (
            target_polyphony_gain - self.polyphony_gain
        ) * poly_alpha

        out *= MASTER_VOICE_GAIN * self.polyphony_gain

        # Optional FX after voices are summed.
        out = self.apply_fx(out)

        # INPUT LEVEL meter: before the user Volume fader.
        input_peak = float(np.max(np.abs(out)) + 1e-8)

        # User volume.
        out *= self.volume

        # Transparent limiter: no tanh distortion, no per-block normalization.
        out = self.apply_transparent_limiter(out)

        # Absolute safety ceiling for unexpected peaks or driver glitches.
        out = np.clip(out, -0.98, 0.98).astype(np.float32)

        # OUTPUT LEVEL meter: after Volume and limiter.
        output_peak = float(np.max(np.abs(out)) + 1e-8)

        self.input_level_peak = max(
            input_peak,
            self.input_level_peak * 0.86,
        )
        self.output_level_peak = max(
            output_peak,
            self.output_level_peak * 0.86,
        )

        # Clip warning is based on the pre-volume signal and final output.
        if input_peak >= 1.0 or output_peak >= 1.0:
            self.clip_detected = True

        outdata[:, 0] = out.astype(np.float32)


class PreviewPlayer:
    """
    Single-voice one-shot preview player (used by "Play Input"/"Play Output").

    sd.play()/sd.stop() cannot switch clips cleanly: starting a new play()
    call stops whatever is currently playing at its literal current sample,
    which is almost never a zero crossing, producing an audible pop. This
    player instead keeps its own persistent OutputStream and, when a new
    clip interrupts one still playing, fades the outgoing clip to silence
    over a few ms before the new clip starts.
    """

    # Long enough to span a full cycle of a low bass fundamental (~65Hz is
    # ~15ms) -- cutting mid-cycle pops even with a smooth amplitude fade.
    FADE_SECONDS = 0.025

    def __init__(self, blocksize=256):
        self.blocksize = blocksize
        self.lock = threading.Lock()
        self.sample_rate = None
        self.stream = None
        self.buffer = np.zeros(0, dtype=np.float32)
        self.pos = 0
        self.pending_buffer = None
        self.fade_samples = 1
        self.fade_remaining = 0

    def _open_stream(self, sample_rate):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

        self.sample_rate = sample_rate
        self.fade_samples = max(1, int(self.FADE_SECONDS * sample_rate))
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, outdata, frames, time, status):
        with self.lock:
            out = np.zeros(frames, dtype=np.float32)
            filled = 0

            while filled < frames:
                avail = len(self.buffer) - self.pos

                if avail <= 0:
                    if self.pending_buffer is not None:
                        self.buffer = self.pending_buffer
                        self.pos = 0
                        self.fade_remaining = 0
                        self.pending_buffer = None
                        continue
                    break

                n = min(frames - filled, avail)
                chunk = self.buffer[self.pos : self.pos + n].astype(
                    np.float32, copy=True
                )

                if self.fade_remaining > 0:
                    k = min(self.fade_remaining, n)
                    start_level = self.fade_remaining / self.fade_samples
                    end_level = (self.fade_remaining - k) / self.fade_samples
                    chunk[:k] *= np.linspace(
                        start_level, end_level, k, endpoint=False, dtype=np.float32
                    )
                    if k < n:
                        chunk[k:] = 0.0
                    self.fade_remaining -= k

                self.pos += n
                out[filled : filled + n] = chunk
                filled += n

                if self.fade_remaining <= 0 and self.pending_buffer is not None:
                    self.buffer = self.pending_buffer
                    self.pos = 0
                    self.pending_buffer = None

            outdata[:, 0] = out

    def play(self, audio, sample_rate):
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        if self.stream is None or self.sample_rate != sample_rate:
            self._open_stream(sample_rate)

        with self.lock:
            currently_playing = self.pos < len(self.buffer)
            if currently_playing:
                if self.fade_remaining <= 0:
                    self.fade_remaining = self.fade_samples
                self.pending_buffer = audio
            else:
                self.buffer = audio
                self.pos = 0
                self.fade_remaining = 0
                self.pending_buffer = None

    def stop(self):
        """
        Fades whatever's currently playing to silence over FADE_SECONDS
        instead of cutting it off mid-sample (see class docstring for why
        that pops) -- for callers that need to silence an active preview
        without starting a replacement clip, e.g. loading a different
        timbre while one is still playing. Leaves the stream itself open
        (unlike close()) so a later play() doesn't pay to reopen it.
        """
        with self.lock:
            if self.pos < len(self.buffer) and self.fade_remaining <= 0:
                self.fade_remaining = self.fade_samples
            self.pending_buffer = None

    def close(self):
        with self.lock:
            self.buffer = np.zeros(0, dtype=np.float32)
            self.pending_buffer = None

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class SynthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Timbre Shadowing")
        self.root.geometry("1320x820")
        self.root.configure(bg="#161616")

        self.wav_by_midi = {}
        self.key_items = {}

        self.min_midi = None
        self.max_midi = None
        self.home_base_midi = 60
        self.base_midi = 60
        self.octave_shift = 0

        self.pressed_keyboard_keys = {}
        self.release_jobs = {}
        self.mouse_down_note = None

        self.engine = None
        self.preview_player = PreviewPlayer()
        # Bumped to invalidate any in-flight preview load (decoded on a
        # worker thread) so it skips play() if something else, e.g. Load,
        # claims audio priority before it finishes -- avoids a pop from two
        # things starting playback at once.
        self.preview_generation = 0
        self.run_process = None
        self.run_thread = None
        self.run_queue = queue.Queue()
        self.run_in_progress = False
        # retrain_classifier_worker runs on a background thread; completion
        # is handed back through this queue and drained by a main-thread
        # poll, since Tcl doesn't reliably service root.after() called from
        # another thread.
        self.retrain_process = None
        self.retrain_queue = queue.Queue()
        self.retrain_queue_after_id = None
        # Ignores any leftover input/target.wav on disk -- Run System stays
        # unavailable until audio is uploaded in this session.
        self.audio_uploaded = False
        self.selected_input_name = "No audio selected"

        # (family_folder, instrument_folder, sound_folder) of the timbre
        # currently loaded, if any -- lets reopening Timbre Manager jump
        # straight back to it.
        self.loaded_timbre_location = None

        # Training-data file the last classification correction produced
        # for the current input, so correcting it again replaces rather
        # than duplicates.
        self._current_input_correction_file = None

        self.tutorial_window = None

        # External MIDI keyboard state.
        self.midi_input = None
        self.midi_window = None
        self.midi_port_var = tk.StringVar(value="")
        self.midi_status_var = tk.StringVar(value="Disconnected")
        self.midi_port_combo = None
        self.midi_held_notes = set()
        # raw physical-key MIDI note -> transposed note actually played,
        # so note_off releases the right note even if home_base_midi
        # changes while the key is still held.
        self.midi_note_transposition = {}
        # A noisy analog pot sends CC1 messages that jitter by 1/127 at
        # rest -- rendered 1:1 that reads as visible shaking, so only
        # re-render past this threshold from the last drawn value.
        self._last_mod_wheel_cc = None
        self.midi_autoconnect_after_id = None
        self.midi_auto_connect_suspended = False
        self.midi_known_ports = set()
        self.midi_scan_in_progress = False
        self.midi_open_in_progress = False
        # mido.get_input_names() can hang on a mid-scan unplug; these let
        # the autoconnect poll detect and abandon a stuck scan.
        self.midi_scan_id = 0
        self.midi_scan_started_at = None
        # One persistent thread serves all MIDI scans (see
        # _ensure_midi_scan_worker) -- a fresh thread per scan was found to
        # eventually stop CoreMIDI from reporting hardware changes.
        self.midi_scan_request_queue = queue.Queue()
        self.midi_scan_worker_generation = 0
        self._midi_scan_worker_alive = False
        # Worker->main-thread handoff via a thread-safe queue, drained by a
        # periodic poll -- Tcl doesn't reliably service root.after() called
        # from another thread.
        self.midi_queue = queue.Queue()
        self.midi_queue_after_id = None
        # Note/CC/pitchwheel messages need much lower latency than the
        # 80ms midi_queue poll would give (laggy live playing), so they get
        # their own queue with a faster dedicated poll.
        self.midi_note_queue = queue.Queue()
        self.midi_note_queue_after_id = None
        self.midi_toast = None
        self.midi_toast_after_id = None

        # Timbre Manager state.
        self.sound_manager_window = None
        self.sound_manager_taxonomy_list = None
        self.sound_manager_sound_list = None
        self.sound_manager_taxonomy_rows = []
        self.sound_manager_sound_records = []
        self.sound_manager_details_label = None

        # Data Overview state.
        self.data_overview_window = None
        self.data_overview_list = None
        self.data_overview_total_label = None

        # Recognition Accuracy state (real-world agreement rate).
        self.accuracy_window = None
        self.accuracy_list = None
        self.accuracy_list_row_instrument = {}

        # Test Set Accuracy state (train_classifier.py's held-out
        # test-set numbers -- a separate, training-time diagnostic).
        self.test_accuracy_window = None
        self.test_accuracy_list = None
        self.test_accuracy_overall_label = None

        # "Correct CNN Label" chooser dialogs -- tracked so a misclick/
        # double-click on the button or the clickable prediction label
        # can't stack two of these on top of each other.
        self._manager_fix_classification_chooser = None
        self._fix_classification_chooser = None
        self.accuracy_overall_label = None

        self.run_button = None
        self.refresh_button = None
        self.play_input_button = None
        self.play_output_button = None
        self.save_sound_button = None
        self.save_timbre_button = None
        self.timbre_ready = False

        # Center loading spinner shown while backend is running.
        self.loading_overlay = None
        self.loading_canvas = None
        self.loading_arc = None
        self.loading_label = None
        self.loading_angle = 0
        self.loading_progress_fill = None
        self.loading_progress_sub_label = None

        self.classifier_retrain_in_progress = False
        self.loading_after_id = None

        self.controls = {}

        self.output_meter = {}
        self.clip_label = None
        self.meter_after_id = None

        # User-configurable pitch bend range, up/down independently; down
        # is stored negative. Must be set before build_ui() runs, since it
        # reads these for the steppers' initial values.
        self.pitch_bend_range_up_semitones = PITCH_BEND_RANGE_SEMITONES
        self.pitch_bend_range_down_semitones = -PITCH_BEND_RANGE_SEMITONES

        self.build_ui()
        self.bind_computer_keyboard()
        self.reset_run_state()
        self.start_midi_autoconnect()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        self.root.geometry("1536x1050")
        self.root.minsize(1180, 700)
        self.root.configure(bg="#111211")
        self.ui_font = "Segoe UI"

        title = tk.Label(
            self.root,
            text="Timbre Shadowing",
            fg="#f4f4f4",
            bg="#171817",
            font=(self.ui_font, 26, "bold"),
        )
        title.pack(fill="x", pady=(6, 4))

        top = tk.Frame(self.root, bg="#111211", height=618)
        top.pack(fill="x", padx=18, pady=(0, 2))
        top.pack_propagate(False)
        top.grid_rowconfigure(0, weight=1)
        for c, weight in enumerate((125, 162, 345, 150)):
            top.grid_columnconfigure(c, weight=weight, uniform="panels")

        self.build_classifier_panel(top)
        self.build_knob_panel(top, "EQ", ["EQ_FREQ", "EQ_BOOST"], 1)
        self.build_knob_panel(top, "ADSR", ["ATTACK", "DECAY", "SUSTAIN", "RELEASE"], 2)
        self.build_output_panel(top, 3)

        status = tk.Frame(self.root, bg="#171817", height=42)
        status.pack(fill="x", padx=18, pady=(0, 7))
        status.pack_propagate(False)
        self.classification_available = False
        self.prediction_label = tk.Label(
            status,
            text="Predicted Class: not loaded",
            fg="#8d8d8d",
            bg="#242017",
            font=(self.ui_font, 13, "bold"),
            padx=10,
            pady=3,
            highlightthickness=1,
        )
        self.prediction_label.pack(side="left", padx=(12, 6))
        self.prediction_label.bind("<Button-1>", self.fix_current_classification)
        self.prediction_label.bind("<Enter>", self._on_prediction_hover_enter)
        self.prediction_label.bind("<Leave>", self._on_prediction_hover_leave)

        self.prediction_fix_hint = tk.Label(
            status,
            text="",
            fg="#8d8d8d",
            bg="#171817",
            font=(self.ui_font, 9, "italic"),
        )
        self.prediction_fix_hint.pack(side="left", padx=(0, 32))
        self.prediction_fix_hint.bind("<Button-1>", self.fix_current_classification)
        self._set_prediction_clickable(False)

        self.count_label = tk.Label(
            status, text="Notes: 0", fg="#aeb2b5", bg="#171817", font=(self.ui_font, 13)
        )
        self.count_label.pack(side="left", padx=(0, 75))
        self.octave_label = tk.Label(
            status,
            text="Base: -- | Octave: --",
            fg="#aeb2b5",
            bg="#171817",
            font=(self.ui_font, 13),
        )
        self.octave_label.pack(side="left")
        self.tutorial_button = self._styled_button(
            status, "Tutorial", self.open_keyboard_tutorial
        )
        self.tutorial_button.pack(side="right", padx=28, pady=3, ipady=2)

        # One shared border around wheels + keyboard so they read as a
        # single panel, instead of the wheels floating outside the
        # keyboard's own box to its left.
        keyboard_row = tk.Frame(
            self.root,
            bg="#171817",
            highlightthickness=1,
            highlightbackground="#676767",
            highlightcolor="#676767",
        )
        keyboard_row.pack(fill="x", padx=18, pady=(0, 4))

        wheels_frame = tk.Frame(keyboard_row, bg="#171817")
        wheels_frame.pack(side="left", fill="y", padx=(8, 6), pady=8)

        # Up/down pitch bend range, always visible and independently
        # adjustable, replacing the earlier click-to-open range dialog.
        range_frame = tk.Frame(wheels_frame, bg="#171817")
        range_frame.pack(side="left", padx=(0, 8), pady=(70, 0))

        tk.Label(
            range_frame,
            text="Bend Range",
            fg="#f2f2f2",
            bg="#171817",
            font=(self.ui_font, 11, "bold"),
        ).pack(pady=(0, 10))

        def on_range_up_change(value):
            self.pitch_bend_range_up_semitones = value
            # Re-apply the currently-held bend amount at the new range so
            # the audio doesn't keep using the old range until next drag,
            # and refresh the fader's live readout to match.
            self.on_pitch_wheel_change(self.pitch_wheel.get())
            self.pitch_wheel.set(self.pitch_wheel.get())

        def on_range_down_change(value):
            self.pitch_bend_range_down_semitones = value
            self.on_pitch_wheel_change(self.pitch_wheel.get())
            self.pitch_wheel.set(self.pitch_wheel.get())

        up_row = tk.Frame(range_frame, bg="#171817")
        up_row.pack(pady=(0, 6))
        tk.Label(
            up_row, text="Up", fg="#8f9295", bg="#171817", font=(self.ui_font, 9)
        ).pack(side="left", padx=(0, 4))
        self.pitch_bend_up_stepper = NumberStepper(
            up_row,
            initial=int(self.pitch_bend_range_up_semitones),
            minimum=1,
            maximum=12,
            command=on_range_up_change,
        )
        self.pitch_bend_up_stepper.pack(side="left")

        down_row = tk.Frame(range_frame, bg="#171817")
        down_row.pack()
        tk.Label(
            down_row, text="Down", fg="#8f9295", bg="#171817", font=(self.ui_font, 9)
        ).pack(side="left", padx=(0, 4))
        self.pitch_bend_down_stepper = NumberStepper(
            down_row,
            initial=int(self.pitch_bend_range_down_semitones),
            minimum=-12,
            maximum=-1,
            command=on_range_down_change,
        )
        self.pitch_bend_down_stepper.pack(side="left")

        # 0..1 normalized (matches CC1's 0-127 range once scaled), stays
        # wherever it's set -- no spring-back, like a real mod wheel.
        self.mod_wheel_widget = MixerFader(
            wheels_frame,
            "Mod",
            initial=0.0,
            min_value=0.0,
            max_value=1.0,
            resolution=0.01,
            command=self.on_mod_wheel_change,
            show_value_text=False,
            width=48,
            # Matches Bend's height/bottom_margin so the two tracks stay
            # aligned side by side.
            height=320,
            track_top=44,
            bottom_margin=80,
            label_y=18,
        )
        self.mod_wheel_widget.pack(side="left", padx=(0, 4))

        def format_pitch_bend_value(normalized_value):
            # Mirrors on_pitch_wheel_change's asymmetric up/down math.
            if normalized_value >= 0:
                semitones = normalized_value * self.pitch_bend_range_up_semitones
            else:
                semitones = -normalized_value * self.pitch_bend_range_down_semitones
            if abs(semitones) < 0.05:
                return "0.0"
            return f"{'+' if semitones > 0 else ''}{semitones:.1f}"

        # -1..+1 normalized, converted to the up/down semitone range only
        # at the boundary into the audio engine. Springs back to 0 on
        # release, like a real pitch wheel, with a live semitone readout.
        self.pitch_wheel = MixerFader(
            wheels_frame,
            "Bend",
            initial=0.0,
            min_value=-1.0,
            max_value=1.0,
            resolution=0.01,
            command=self.on_pitch_wheel_change,
            value_formatter=format_pitch_bend_value,
            width=48,
            # Taller than Mod's original size so the value text at the
            # bottom of travel has clearance without shrinking the track.
            height=320,
            track_top=44,
            bottom_margin=80,
            label_y=18,
            spring_back_to=0.0,
        )
        self.pitch_wheel.pack(side="left")

        self.keyboard_canvas = tk.Canvas(
            keyboard_row,
            # The keyboard drawing uses 225 px white keys plus top/bottom spacing.
            # 175 px clipped the lower part of the keys and made them look too short.
            height=270,
            bg="#171817",
            highlightthickness=0,
            bd=0,
        )
        self.keyboard_canvas.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        self.keyboard_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.keyboard_canvas.bind("<ButtonRelease-1>", self.on_mouse_release_anywhere)
        self.keyboard_canvas.bind("<Leave>", self.on_mouse_leave_keyboard)
        self.keyboard_canvas.bind(
            "<Configure>",
            lambda event: self.draw_keyboard() if self.wav_by_midi else None,
        )

        # Not shown -- messages still recorded via self.log() for error
        # dialogs/debugging, just not surfaced here for normal use.
        log_box = self._bordered_panel(self.root, "LOG", bg="#151615")
        log_toolbar = tk.Frame(log_box, bg="#151615")
        log_toolbar.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(
            log_toolbar,
            text="System messages and backend progress",
            fg="#8f9295",
            bg="#151615",
            font=(self.ui_font, 9),
        ).pack(side="left")
        tk.Button(
            log_toolbar,
            text="Clear",
            command=lambda: self.log_text.delete("1.0", tk.END),
            bg="#2b2c2d",
            fg="#eeeeee",
            activebackground="#3a3b3c",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=12,
            pady=2,
            cursor="hand2",
        ).pack(side="right")
        log_body = tk.Frame(log_box, bg="#0d0e0d")
        log_body.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        log_scrollbar = tk.Scrollbar(log_body)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text = tk.Text(
            log_body,
            height=6,
            bg="#0d0e0d",
            fg="#d8d8d8",
            insertbackground="#ffffff",
            selectbackground="#385b77",
            relief="flat",
            bd=0,
            wrap="word",
            font=("Consolas", 9),
            yscrollcommand=log_scrollbar.set,
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.config(command=self.log_text.yview)

    def show_loading(self, message="Running system..."):
        if self.loading_overlay is not None:
            return

        self.loading_angle = 0

        self.loading_overlay = tk.Frame(
            self.root,
            bg="#111111",
            bd=2,
            relief="ridge",
        )

        # Center of window. Covers only a compact dialog-like area.
        self.loading_overlay.place(
            relx=0.5,
            rely=0.48,
            anchor="center",
            width=280,
            height=220,
        )
        self.loading_overlay.lift()

        self.loading_canvas = tk.Canvas(
            self.loading_overlay,
            width=90,
            height=90,
            bg="#111111",
            highlightthickness=0,
        )
        self.loading_canvas.pack(pady=(22, 8))

        self.loading_canvas.create_oval(
            12,
            12,
            78,
            78,
            outline="#333333",
            width=8,
        )

        self.loading_arc = self.loading_canvas.create_arc(
            12,
            12,
            78,
            78,
            start=0,
            extent=95,
            style="arc",
            outline="#f8d36a",
            width=8,
        )

        self.loading_label = tk.Label(
            self.loading_overlay,
            text=message,
            fg="#eeeeee",
            bg="#111111",
            font=("Arial", 13, "bold"),
        )
        self.loading_label.pack(pady=(8, 2))

        # ttk.Progressbar renders native macOS blue regardless of styling --
        # a plain tk.Frame trough + fill reliably stays gold instead.
        progress_trough = tk.Frame(
            self.loading_overlay,
            bg="#2b2c2d",
            width=220,
            height=10,
            highlightthickness=1,
            highlightbackground="#3a3b3c",
        )
        progress_trough.pack(pady=(2, 4))
        progress_trough.pack_propagate(False)

        self.loading_progress_fill = tk.Frame(progress_trough, bg="#f8d36a")
        self.loading_progress_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        self.loading_progress_sub_label = tk.Label(
            self.loading_overlay,
            text="Training / rendering notes...",
            fg="#bbbbbb",
            bg="#111111",
            font=("Arial", 10),
        )
        self.loading_progress_sub_label.pack()

        self.animate_loading()

    def animate_loading(self):
        if self.loading_overlay is None or self.loading_canvas is None:
            return

        self.loading_angle = (self.loading_angle + 12) % 360

        if self.loading_arc is not None:
            self.loading_canvas.itemconfig(
                self.loading_arc,
                start=self.loading_angle,
            )

        self.loading_after_id = self.root.after(45, self.animate_loading)

    def hide_loading(self):
        if self.loading_after_id is not None:
            try:
                self.root.after_cancel(self.loading_after_id)
            except Exception:
                pass

        self.loading_after_id = None

        if self.loading_overlay is not None:
            try:
                self.loading_overlay.destroy()
            except Exception:
                pass

        self.loading_overlay = None
        self.loading_canvas = None
        self.loading_arc = None
        self.loading_label = None
        self.loading_progress_fill = None
        self.loading_progress_sub_label = None

    def show_toast(self, message, duration_ms=3000, fg="#58c96b"):
        """Brief, non-blocking notification in the corner of the main
        window -- unlike messagebox, doesn't grab focus or require a click
        to dismiss. Used for background events like a MIDI device
        connecting, where a modal popup would be more interruption than
        the event deserves."""
        if self.midi_toast_after_id is not None:
            try:
                self.root.after_cancel(self.midi_toast_after_id)
            except Exception:
                pass
            self.midi_toast_after_id = None

        if self.midi_toast is not None:
            try:
                self.midi_toast.destroy()
            except Exception:
                pass
            self.midi_toast = None

        self.midi_toast = tk.Label(
            self.root,
            text=message,
            fg=fg,
            bg="#1c1d1c",
            font=(self.ui_font, 10, "bold"),
            padx=14,
            pady=8,
            highlightthickness=1,
            highlightbackground="#3a3b3c",
        )
        self.midi_toast.place(relx=1.0, rely=0.0, x=-14, y=14, anchor="ne")
        self.midi_toast.lift()

        def dismiss():
            self.midi_toast_after_id = None
            if self.midi_toast is not None:
                try:
                    self.midi_toast.destroy()
                except Exception:
                    pass
                self.midi_toast = None

        self.midi_toast_after_id = self.root.after(duration_ms, dismiss)

    def update_loading_progress(self, phase_text, percent=None, sub_text=None):
        if self.loading_label is not None:
            self.loading_label.config(text=phase_text)

        if percent is not None and self.loading_progress_fill is not None:
            self.loading_progress_fill.place(relwidth=max(0, min(100, percent)) / 100)

        if sub_text is not None and self.loading_progress_sub_label is not None:
            self.loading_progress_sub_label.config(text=sub_text)

    def build_classifier_panel(self, parent):
        box = self._bordered_panel(parent, "INPUT")
        box.grid(row=0, column=0, padx=(0, 8), pady=2, sticky="nsew")

        self.drop_zone = tk.Label(
            box,
            text=self._drop_zone_default_text(),
            fg="#eeeeee",
            bg="#171b20",
            relief="solid",
            bd=1,
            justify="center",
            wraplength=180,
            font=(self.ui_font, 10),
            cursor="hand2",
        )
        self.drop_zone.pack(fill="x", padx=12, pady=(26, 24), ipady=28)
        self.drop_zone.bind("<Button-1>", self.browse_for_audio_file)

        if DND_FILES is not None:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<DropEnter>>", self.on_drop_enter)
            self.drop_zone.dnd_bind("<<DropLeave>>", self.on_drop_leave)
            self.drop_zone.dnd_bind("<<Drop>>", self.on_audio_drop)
        else:
            self.drop_zone.config(
                text="Drag & drop unavailable\nInstall tkinterdnd2", fg="#f8d36a"
            )

        def action_button(text, command):
            button = self._styled_button(box, text, command)
            button.pack(fill="x", padx=13, pady=2, ipady=2)
            return button

        self._section_divider(box, "SYSTEM")
        self.run_button = action_button("Run System", self.run_system_threaded)
        self._set_button_enabled(self.run_button, self.audio_uploaded)
        action_button("MIDI Controller", self.open_midi_manager)
        self.refresh_button = action_button("Refresh", self.discard_generated_notes)
        self._set_button_enabled(self.refresh_button, self.audio_uploaded)

        self._section_divider(box, "TIMBRE")
        action_button("Timbre Manager", self.open_sound_manager)
        self.save_timbre_button = action_button(
            "Save Timbre", self.save_timbre_from_main
        )
        self._set_button_enabled(self.save_timbre_button, self.timbre_ready)

        self._section_divider(box, "DATA")
        action_button("Data Overview", self.open_data_overview)
        action_button("Test Set Accuracy", self.open_test_accuracy_overview)
        action_button("Recognition Accuracy", self.open_accuracy_overview)

        self._section_divider(box, "DANGER ZONE")
        reset_button = self._danger_button(box, "Reset Whole System", self.reset_whole_system)
        reset_button.pack(fill="x", padx=13, pady=2, ipady=2)

    def open_midi_manager(self):
        if self.midi_window is not None and self.midi_window.winfo_exists():
            self.midi_window.deiconify()
            self.midi_window.lift()
            self.midi_window.focus_force()
            self.refresh_midi_ports()
            return

        self.midi_window = tk.Toplevel(self.root)
        self.midi_window.title("MIDI Controller")
        self.midi_window.geometry("560x245")
        self.midi_window.resizable(False, False)
        self.midi_window.configure(bg="#181918")
        self.midi_window.transient(self.root)
        self.midi_window.protocol("WM_DELETE_WINDOW", self.close_midi_manager_window)
        self._center_toplevel(self.midi_window)

        title = tk.Label(
            self.midi_window,
            text="External MIDI Controller",
            fg="#f3f3f3",
            bg="#181918",
            font=(self.ui_font, 17, "bold"),
        )
        title.pack(pady=(18, 12))

        device_frame = tk.Frame(self.midi_window, bg="#181918")
        device_frame.pack(fill="x", padx=24)

        tk.Label(
            device_frame,
            text="MIDI Input:",
            fg="#dddddd",
            bg="#181918",
            font=(self.ui_font, 10, "bold"),
        ).pack(side="left", padx=(0, 8))

        self.midi_port_combo = ttk.Combobox(
            device_frame,
            textvariable=self.midi_port_var,
            # Starts disabled (no devices known yet, before the first
            # async scan completes) rather than readonly-with-nothing-
            # to-select -- see the comment in _on_midi_scan_done for why
            # an empty readonly dropdown is worth avoiding.
            state="disabled",
            width=43,
        )
        self.midi_port_combo.pack(side="left", fill="x", expand=True)

        button_frame = tk.Frame(self.midi_window, bg="#181918")
        button_frame.pack(fill="x", padx=24, pady=(18, 10))

        def midi_button(label, command):
            button = self._styled_button(button_frame, label, command)
            button.config(font=(self.ui_font, 10), padx=13, pady=5)
            button.pack(side="left", padx=(0, 9))
            return button

        midi_button("Refresh Devices", self.refresh_midi_ports)
        midi_button("Connect", self.connect_midi_input)
        midi_button(
            "Disconnect", lambda: self.disconnect_midi_input(user_initiated=True)
        )

        status_frame = tk.Frame(self.midi_window, bg="#101110", bd=1, relief="solid")
        status_frame.pack(fill="x", padx=24, pady=(4, 8), ipady=9)

        tk.Label(
            status_frame,
            text="Status:",
            fg="#aaaaaa",
            bg="#101110",
            font=(self.ui_font, 10, "bold"),
        ).pack(side="left", padx=(12, 6))

        tk.Label(
            status_frame,
            textvariable=self.midi_status_var,
            fg="#58c96b",
            bg="#101110",
            font=(self.ui_font, 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        tk.Label(
            self.midi_window,
            text=(
                "Keyboards connect automatically. Use Connect/Disconnect to "
                "switch or silence a device.\n"
                "Supports velocity, volume, pitch bend, and mod wheel."
            ),
            fg="#929292",
            bg="#181918",
            font=(self.ui_font, 9),
            justify="left",
            wraplength=512,
        ).pack(fill="x", padx=24, pady=(2, 0))

        self.refresh_midi_ports()

    def close_midi_manager_window(self):
        if self.midi_window is not None:
            try:
                self.midi_window.destroy()
            except Exception:
                pass
        self.midi_window = None
        self.midi_port_combo = None

    def open_keyboard_tutorial(self):
        if self.tutorial_window is not None and self.tutorial_window.winfo_exists():
            self.tutorial_window.deiconify()
            self.tutorial_window.lift()
            self.tutorial_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self.tutorial_window = window
        window.title("Controls Guide")
        window.geometry("1000x920")
        window.minsize(700, 480)
        window.configure(bg="#151615")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self.close_keyboard_tutorial_window)
        self._center_toplevel(window)

        tk.Label(
            window,
            text="Controls Guide",
            fg="#f4f4f4",
            bg="#171817",
            font=(self.ui_font, 18, "bold"),
        ).pack(fill="x", pady=(0, 10), ipady=10)

        # Scrollable so this can keep growing with more sections without
        # being capped by the screen's actual usable height.
        scroll_wrap = tk.Frame(window, bg="#151615")
        scroll_wrap.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        scroll_canvas = tk.Canvas(scroll_wrap, bg="#151615", highlightthickness=0)
        scrollbar = tk.Scrollbar(
            scroll_wrap, orient="vertical", command=scroll_canvas.yview
        )
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(scroll_canvas, bg="#151615")
        body_window = scroll_canvas.create_window((0, 0), window=body, anchor="nw")

        # Content is static, so scrollregion is computed once (explicit
        # sync below), not rebound to body's <Configure> -- that would
        # feedback-loop with scroll_canvas's own <Configure> and peg a CPU
        # core.
        geometry_job = [None]

        def apply_scroll_geometry():
            geometry_job[0] = None
            scroll_canvas.itemconfig(body_window, width=scroll_canvas.winfo_width())
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def schedule_scroll_geometry(_event=None):
            if geometry_job[0] is not None:
                scroll_canvas.after_cancel(geometry_job[0])
            geometry_job[0] = scroll_canvas.after(10, apply_scroll_geometry)

        scroll_canvas.bind("<Configure>", schedule_scroll_geometry)

        def on_mousewheel(event):
            # event.delta is multiples of 120 on Windows, but macOS Aqua
            # gives small raw values -- dividing those by 120 truncates to
            # 0 and silently does nothing.
            if sys.platform == "darwin":
                step = -1 if event.delta > 0 else 1
            else:
                step = int(-1 * (event.delta / 120))
            scroll_canvas.yview_scroll(step, "units")

        def bind_mousewheel(_event=None):
            scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(_event=None):
            scroll_canvas.unbind_all("<MouseWheel>")

        scroll_canvas.bind("<Enter>", bind_mousewheel)
        scroll_canvas.bind("<Leave>", unbind_mousewheel)

        def section(title, subtitle=None):
            tk.Label(
                body,
                text=title,
                fg="#f8d36a",
                bg="#151615",
                font=(self.ui_font, 12, "bold"),
                anchor="w",
            ).pack(fill="x", pady=(14, 0))
            if subtitle:
                tk.Label(
                    body,
                    text=subtitle,
                    fg="#aeb2b5",
                    bg="#151615",
                    font=(self.ui_font, 9),
                    justify="left",
                    wraplength=500,
                    anchor="w",
                ).pack(fill="x", pady=(2, 4))

        def key_row(pairs):
            """pairs: list of (key label, note name) tuples, drawn as a
            row of small piano-key-style badges."""
            row = tk.Frame(body, bg="#151615")
            row.pack(fill="x", pady=(2, 0))
            for key_label, note_label in pairs:
                cell = tk.Frame(
                    row,
                    bg="#0d0e0d",
                    highlightthickness=1,
                    highlightbackground="#3a3b3c",
                )
                cell.pack(side="left", padx=3, ipadx=6, ipady=4)
                tk.Label(
                    cell,
                    text=key_label,
                    fg="#eeeeee",
                    bg="#0d0e0d",
                    font=(self.ui_font, 11, "bold"),
                ).pack()
                tk.Label(
                    cell,
                    text=note_label,
                    fg="#4eb7ff",
                    bg="#0d0e0d",
                    font=(self.ui_font, 9),
                ).pack()

        section(
            "Playable Keys",
            "Type on your computer keyboard like a piano -- each key below "
            "is labeled with the computer key that plays it.",
        )
        tutorial_keyboard_canvas = tk.Canvas(body, bg="#151615", highlightthickness=0)
        tutorial_keyboard_canvas.pack(pady=(6, 0))
        self.draw_tutorial_keyboard(tutorial_keyboard_canvas)

        section(
            "Octave Shift",
            "Shifts which octave the keys above play, snapped to the "
            "nearest octave Run System actually rendered -- it's clamped "
            "so you can't shift past that range.",
        )
        key_row([("Z", "Octave down"), ("X", "Octave up")])

        section(
            "EQ",
            "FREQ picks which frequency to shape; GAIN boosts (up) or "
            "cuts (down) the sound at that frequency. The curve here "
            "shows an example boost, with a matching cut underneath it "
            "for reference.",
        )
        eq_diagram_canvas = tk.Canvas(body, bg="#151615", highlightthickness=0)
        eq_diagram_canvas.pack(pady=(6, 0))
        self.draw_eq_diagram(eq_diagram_canvas)

        section(
            "ADSR Envelope",
            "Shapes how a note's volume changes over time, from the "
            "moment you press a key to when it fades to silence.",
        )
        adsr_diagram_canvas = tk.Canvas(body, bg="#151615", highlightthickness=0)
        adsr_diagram_canvas.pack(pady=(6, 0))
        self.draw_adsr_diagram(adsr_diagram_canvas)

        section(
            "External MIDI Controller",
            'Prefer a real keyboard? Click "MIDI Controller" in the left '
            "panel to connect one and play the same notes on it instead, "
            "including its pitch bend and mod wheel.",
        )

        section(
            "Pitch Bend & Mod Wheel",
            "The two wheels beside the on-screen keyboard work with the "
            "mouse too. PITCH bends the currently playing notes up or "
            "down and springs back to center when released, like a real "
            "pitch wheel. MOD adds vibrato and stays wherever you leave "
            "it.",
        )

        # One-time initial geometry sync now that all (static) content has
        # been added -- see the comment above schedule_scroll_geometry.
        window.update_idletasks()
        scroll_canvas.itemconfig(body_window, width=scroll_canvas.winfo_width())
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        tk.Button(
            window,
            text="Got it",
            command=self.close_keyboard_tutorial_window,
            bg="#f8d36a",
            fg="#111111",
            activebackground="#ffdf8c",
            relief="flat",
            font=(self.ui_font, 10, "bold"),
            cursor="hand2",
        ).pack(fill="x", padx=20, pady=(0, 16), ipady=6)

    def close_keyboard_tutorial_window(self):
        if self.tutorial_window is not None:
            try:
                self.tutorial_window.destroy()
            except Exception:
                pass
        self.tutorial_window = None

    def draw_tutorial_keyboard(self, canvas):
        """
        Draw a mini piano matching the app's real on-screen keyboard, but
        labeled with the computer keys (from KEY_TO_OFFSET) instead of MIDI
        note numbers, so it stays correct if the keymap ever changes.
        """
        offset_to_key = {offset: key for key, offset in KEY_TO_OFFSET.items()}
        display_key = {"semicolon": ";", "apostrophe": "'", "bracketleft": "["}

        def key_label(offset):
            raw = offset_to_key[offset]
            return display_key.get(raw, raw.upper())

        white_offsets = sorted(
            offset for offset in offset_to_key if offset % 12 in WHITE_PITCH_CLASSES
        )
        black_offsets = sorted(
            offset for offset in offset_to_key if offset % 12 in BLACK_PITCH_CLASSES
        )

        white_w, white_h = 52, 160
        black_w, black_h = 32, 100
        x_offset, y_offset = 6, 10

        canvas_width = x_offset * 2 + white_w * len(white_offsets)
        canvas_height = y_offset * 2 + white_h
        canvas.config(width=canvas_width, height=canvas_height)

        white_x = {}
        x = x_offset
        for offset in white_offsets:
            y0, y1 = y_offset, y_offset + white_h
            canvas.create_rectangle(
                x, y0, x + white_w, y1, fill="#ffffff", outline="#111111", width=2
            )
            canvas.create_text(
                x + white_w / 2,
                y1 - 24,
                text=key_label(offset),
                fill="#111111",
                font=("Arial", 13, "bold"),
            )
            white_x[offset] = x
            x += white_w

        for offset in black_offsets:
            prev_white = offset - 1
            while prev_white % 12 not in WHITE_PITCH_CLASSES:
                prev_white -= 1
            if prev_white not in white_x:
                continue

            x_center = white_x[prev_white] + white_w
            x0, x1 = x_center - black_w / 2, x_center + black_w / 2
            y0, y1 = y_offset, y_offset + black_h
            canvas.create_rectangle(
                x0, y0, x1, y1, fill="#050505", outline="#222222", width=2
            )
            canvas.create_text(
                (x0 + x1) / 2,
                y1 - 22,
                text=key_label(offset),
                fill="#eeeeee",
                font=("Arial", 11, "bold"),
            )

    def draw_eq_diagram(self, canvas):
        """
        Frequency-response curve showing what the two EQ knobs actually
        do to the sound: FREQ slides the bump left/right (which frequency
        gets affected), GAIN controls how tall a boost or how deep a cut
        it makes there. Axis positions reuse the real EQ_FREQ/EQ_BOOST
        scale marks so the picture matches the actual control, and it's
        a real peaking-EQ bell shape, matching peaking_eq_coefficients().
        """
        width, height = 560, 240
        canvas.config(width=width, height=height)

        left_pad, right_pad = 46, 20
        top_pad, bottom_pad = 14, 56
        plot_w = width - left_pad - right_pad
        plot_h = height - top_pad - bottom_pad
        zero_db_y = top_pad + plot_h / 2

        def x_for_freq_norm(norm):
            return left_pad + (norm / 100.0) * plot_w

        def y_for_db(db):
            return zero_db_y - (db / 12.0) * (plot_h / 2)

        canvas.create_line(
            left_pad,
            zero_db_y,
            left_pad + plot_w,
            zero_db_y,
            fill="#3a3b3c",
            width=1,
        )
        canvas.create_line(
            left_pad,
            top_pad,
            left_pad + plot_w,
            top_pad,
            fill="#242524",
            dash=(2, 3),
        )
        canvas.create_line(
            left_pad,
            top_pad + plot_h,
            left_pad + plot_w,
            top_pad + plot_h,
            fill="#242524",
            dash=(2, 3),
        )

        for norm, label in control_scale_marks("EQ_FREQ"):
            x = x_for_freq_norm(norm)
            canvas.create_line(
                x, top_pad + plot_h, x, top_pad + plot_h + 4, fill="#5a5d60"
            )
            canvas.create_text(
                x,
                top_pad + plot_h + 7,
                text=label,
                fill="#8d8d8d",
                font=(self.ui_font, 7),
                anchor="n",
            )

        for db, label in control_scale_marks("EQ_BOOST"):
            y = y_for_db(db)
            canvas.create_text(
                left_pad - 8,
                y,
                text=label,
                fill="#8d8d8d",
                font=(self.ui_font, 8),
                anchor="e",
            )

        center_norm = log_frequency_to_normalized(1000.0)
        spread = 16.0
        peak_db = 9.0

        def bell(norm, peak):
            t = (norm - center_norm) / spread
            return peak * (2.718281828 ** (-(t * t)))

        samples = 40
        boost_flat = []
        cut_flat = []
        for i in range(samples + 1):
            norm = (i / samples) * 100.0
            boost_flat.extend((x_for_freq_norm(norm), y_for_db(bell(norm, peak_db))))
            cut_flat.extend(
                (x_for_freq_norm(norm), y_for_db(bell(norm, -peak_db * 0.7)))
            )

        canvas.create_line(*cut_flat, fill="#5a5d60", width=2, dash=(4, 2), smooth=True)
        canvas.create_line(*boost_flat, fill="#f8d36a", width=3, smooth=True)

        peak_x = x_for_freq_norm(center_norm)
        peak_y = y_for_db(peak_db)
        canvas.create_line(
            peak_x,
            zero_db_y,
            peak_x,
            peak_y,
            fill="#eeeeee",
            width=1,
            arrow="both",
            arrowshape=(6, 7, 3),
        )
        canvas.create_text(
            peak_x + 8,
            (zero_db_y + peak_y) / 2,
            text="GAIN",
            fill="#eeeeee",
            font=(self.ui_font, 9, "bold"),
            anchor="w",
        )

        freq_arrow_y = top_pad + plot_h + 24
        canvas.create_line(
            peak_x,
            top_pad + plot_h,
            peak_x,
            freq_arrow_y - 6,
            fill="#5a5d60",
            dash=(2, 2),
        )
        canvas.create_line(
            peak_x - 22,
            freq_arrow_y,
            peak_x + 22,
            freq_arrow_y,
            fill="#eeeeee",
            width=1,
            arrow="both",
            arrowshape=(6, 7, 3),
        )
        canvas.create_text(
            peak_x,
            freq_arrow_y + 10,
            text="FREQ",
            fill="#eeeeee",
            font=(self.ui_font, 9, "bold"),
            anchor="n",
        )

    def draw_adsr_diagram(self, canvas):
        """
        Four mini diagrams (Attack/Decay/Sustain/Release), each showing the
        same full envelope shape with just that one phase highlighted --
        the same idea as a standard ADSR reference chart.
        """
        box_w, box_h = 138, 100
        gap = 18
        label_gap = 12
        desc_h = 72
        total_w = box_w * 4 + gap * 3
        total_h = box_h + label_gap + desc_h
        canvas.config(width=total_w, height=total_h)

        # Shared envelope shape, normalized 0-1 within a box (0,0 = top
        # left, 1,1 = bottom right) -- attack rises to a peak, decay falls
        # to the sustain plateau, sustain holds flat, release falls to
        # silence.
        p_start = (0.0, 1.0)
        p_peak = (0.22, 0.0)
        p_decay_end = (0.44, 0.55)
        p_sustain_end = (0.74, 0.55)
        p_end = (1.0, 1.0)
        all_points = [p_start, p_peak, p_decay_end, p_sustain_end, p_end]

        phases = [
            (
                "Attack",
                "#e2725b",
                (p_start, p_peak),
                "When you first press a key, Attack is "
                "the time it takes your sound to go from "
                "silent to the loudest level.",
            ),
            (
                "Decay",
                "#f8d36a",
                (p_peak, p_decay_end),
                "Decay is how long it takes your sound "
                "to fall from that initial peak down to "
                "the sustain level.",
            ),
            (
                "Sustain",
                "#6fcf7d",
                (p_decay_end, p_sustain_end),
                "Sustain is the level held for as long "
                "as you keep the key down -- not a time, "
                "a volume.",
            ),
            (
                "Release",
                "#b39ddb",
                (p_sustain_end, p_end),
                "Release is how long it takes your sound "
                "to fade back to silence after you let "
                "go of the key.",
            ),
        ]

        for i, (name, color, (seg_start, seg_end), desc) in enumerate(phases):
            x0 = i * (box_w + gap)

            def to_canvas(pt, x0=x0):
                px, py = pt
                return x0 + 10 + px * (box_w - 20), 10 + py * (box_h - 20)

            canvas.create_rectangle(
                x0, 0, x0 + box_w, box_h, outline="#eeeeee", width=1
            )

            for pt in (p_peak, p_decay_end):
                cx, _ = to_canvas(pt)
                canvas.create_line(cx, 8, cx, box_h - 8, fill="#5a5d60", width=1)

            flat_points = [coord for pt in all_points for coord in to_canvas(pt)]
            canvas.create_line(*flat_points, fill="#8d8d8d", width=2, joinstyle="round")

            sx0, sy0 = to_canvas(seg_start)
            sx1, sy1 = to_canvas(seg_end)
            canvas.create_line(
                sx0, sy0, sx1, sy1, fill=color, width=3, capstyle="round"
            )

            canvas.create_text(
                x0 + 2,
                box_h + label_gap,
                text=name,
                fill="#f4f4f4",
                font=(self.ui_font, 12, "bold"),
                anchor="nw",
                width=box_w,
            )
            canvas.create_text(
                x0 + 2,
                box_h + label_gap + 22,
                text=desc,
                fill="#aeb2b5",
                font=(self.ui_font, 8),
                anchor="nw",
                width=box_w - 4,
                justify="left",
            )

    # ---- MIDI device access always happens off the Tk main thread -------
    # RtMidi's CoreMIDI backend corrupts the Python/Tcl bridge if scanning/
    # opening ports runs inside a Tk callback -- every mido call below runs
    # on a background thread, marshaled back via root.after(0, ...).

    def refresh_midi_ports(self):
        if mido is None:
            self.midi_status_var.set(
                "MIDI unavailable — install mido and python-rtmidi"
            )
            if self.midi_port_combo is not None:
                self.midi_port_combo["values"] = []
                self.midi_port_combo.config(state="disabled")
            return

        if getattr(self, "midi_scan_in_progress", False):
            return
        self._start_midi_scan(update_combo=True)

    def _start_midi_scan(self, update_combo):
        self.midi_scan_in_progress = True
        self.midi_scan_started_at = time.monotonic()
        self.midi_scan_id += 1
        self._ensure_midi_scan_worker()
        self.midi_scan_request_queue.put((update_combo, self.midi_scan_id))

    def _ensure_midi_scan_worker(self):
        """Starts the persistent scan worker thread if it isn't already
        running for the current generation. A no-op once it's up;
        poll_midi_autoconnect's watchdog bumps midi_scan_worker_generation
        and calls this again to replace a worker stuck in a hung scan.

        Each generation gets its own request queue rather than sharing one
        -- otherwise a request meant for the new worker could be picked up
        by an old, already-blocked worker from a just-retired generation
        and silently dropped (generation mismatch).
        """
        if getattr(self, "_midi_scan_worker_alive", False):
            return
        self._midi_scan_worker_alive = True
        generation = self.midi_scan_worker_generation
        request_queue = queue.Queue()
        self.midi_scan_request_queue = request_queue
        threading.Thread(
            target=self._midi_scan_worker_loop,
            args=(generation, request_queue),
            daemon=True,
            name=f"midi-scan-worker-gen{generation}",
        ).start()

    def _midi_scan_worker_loop(self, generation, request_queue):
        while self.midi_scan_worker_generation == generation:
            update_combo, scan_id = request_queue.get()
            if self.midi_scan_worker_generation != generation:
                break
            self._perform_one_midi_scan(update_combo, scan_id)
        self._midi_scan_worker_alive = False

    def _perform_one_midi_scan(self, update_combo, scan_id):
        # `except ... as exc` deletes exc after the block ends -- a
        # separately-named variable avoids UnboundLocalError referencing
        # it afterward.
        port_names = None
        error = None
        try:
            port_names = list(mido.get_input_names())
        except Exception as exc:
            error = exc
        self.midi_queue.put(("scan_done", port_names, error, update_combo, scan_id))

    def _on_midi_scan_done(self, port_names, error, update_combo, scan_id):
        if scan_id != self.midi_scan_id:
            # A stale result from a scan the watchdog already gave up on
            # (see poll_midi_autoconnect) -- a newer scan may already be
            # running or finished, so this one no longer reflects reality.
            return
        self.midi_scan_in_progress = False

        if error is not None:
            self.midi_status_var.set(f"Device scan failed: {error}")
            self.log(f"MIDI device scan error: {error}")
            return

        if update_combo:
            if self.midi_port_combo is not None:
                self.midi_port_combo["values"] = port_names
                # A readonly Combobox with zero values still opens an
                # empty dropdown on click, swallowing the first click that
                # should've dismissed it -- disable it instead when empty.
                self.midi_port_combo.config(
                    state="readonly" if port_names else "disabled"
                )

            current = self.midi_port_var.get()
            if current not in port_names:
                self.midi_port_var.set(port_names[0] if port_names else "")

            if self.midi_input is None:
                if port_names and self.midi_auto_connect_suspended:
                    self.midi_status_var.set(
                        f"Found {len(port_names)} device(s). Click Connect to resume."
                    )
                elif port_names:
                    self.midi_status_var.set(
                        f"Found {len(port_names)} device(s). Connecting..."
                    )
                else:
                    self.midi_status_var.set("No MIDI input devices found")

        self._process_midi_port_list(port_names)

    def connect_midi_input(self):
        if mido is None:
            messagebox.showerror(
                "MIDI dependency missing",
                "Install MIDI support with:\n\n"
                "python -m pip install mido python-rtmidi",
                parent=self.midi_window or self.root,
            )
            return

        port_name = self.midi_port_var.get().strip()
        if not port_name:
            messagebox.showerror(
                "No MIDI device",
                "Connect a MIDI keyboard, refresh the device list, and select it.",
                parent=self.midi_window or self.root,
            )
            return

        # A manual Connect always overrides any earlier manual Disconnect.
        self.midi_auto_connect_suspended = False
        self._connect_midi_port(port_name, show_errors=True)

    def _connect_midi_port(self, port_name, show_errors=False):
        """Shared connect logic used by both the manual Connect button and
        the background auto-connect poll. The actual mido.open_input()
        call happens in a worker thread -- see the module note above."""
        if getattr(self, "midi_open_in_progress", False):
            return
        self.disconnect_midi_input(update_status=False)
        self.midi_open_in_progress = True
        threading.Thread(
            target=self._open_midi_port_worker,
            args=(port_name, show_errors),
            daemon=True,
        ).start()

    def _open_midi_port_worker(self, port_name, show_errors):
        # See _perform_one_midi_scan's comment -- same
        # except-as-name-gets-deleted pitfall, same fix.
        port = None
        error = None
        try:
            port = mido.open_input(port_name, callback=self.handle_midi_message)
        except Exception as exc:
            error = exc
        self.midi_queue.put(("open_done", port_name, port, error, show_errors))

    def _on_midi_open_done(self, port_name, port, error, show_errors):
        self.midi_open_in_progress = False

        if error is not None:
            self.midi_input = None
            self.midi_status_var.set(f"Connection failed: {error}")
            self.log(f"MIDI connection error: {error}")
            if show_errors:
                messagebox.showerror(
                    "MIDI connection failed",
                    str(error),
                    parent=self.midi_window or self.root,
                )
            return

        self.midi_input = port
        self.midi_status_var.set(f"Connected: {port_name}")
        self.midi_port_var.set(port_name)
        self.log(f"MIDI connected: {port_name}")
        self.show_toast(f"MIDI connected: {port_name}")

        if self.midi_window is not None and self.midi_window.winfo_exists():
            self.refresh_midi_ports()

    def disconnect_midi_input(self, update_status=True, user_initiated=False):
        self.release_all_midi_notes()

        if self.midi_input is not None:
            try:
                self.midi_input.close()
            except Exception as exc:
                self.log(f"MIDI close warning: {exc}")

        self.midi_input = None

        # A user clicking Disconnect means "leave this alone" -- don't let
        # the auto-connect poll immediately grab the same device back.
        # It's re-armed as soon as the physical device list changes again
        # (unplug/replug) or the user clicks Connect.
        if user_initiated:
            self.midi_auto_connect_suspended = True

        if update_status:
            self.midi_status_var.set("Disconnected")
            if hasattr(self, "log_text"):
                self.log("MIDI disconnected.")

    def start_midi_autoconnect(self):
        if self.midi_queue_after_id is None:
            self.poll_midi_queue()
        if self.midi_note_queue_after_id is None:
            self.poll_midi_note_queue()
        if self.midi_autoconnect_after_id is None:
            self.poll_midi_autoconnect()

    def poll_midi_queue(self):
        try:
            while True:
                message = self.midi_queue.get_nowait()
                kind = message[0]
                if kind == "scan_done":
                    _, port_names, error, update_combo, scan_id = message
                    self._on_midi_scan_done(port_names, error, update_combo, scan_id)
                elif kind == "open_done":
                    _, port_name, port, error, show_errors = message
                    self._on_midi_open_done(port_name, port, error, show_errors)
        except queue.Empty:
            pass

        self.midi_queue_after_id = self.root.after(80, self.poll_midi_queue)

    # A real scan (mido.get_input_names()) returns near-instantly -- this
    # is generous headroom, not a tuned expected duration.
    MIDI_SCAN_WATCHDOG_SECONDS = 5.0

    def poll_midi_autoconnect(self):
        if mido is not None:
            if (
                self.midi_scan_in_progress
                and self.midi_scan_started_at is not None
                and time.monotonic() - self.midi_scan_started_at
                > self.MIDI_SCAN_WATCHDOG_SECONDS
            ):
                # Scan worker presumably stuck in mido.get_input_names() --
                # bump both ids to discard its result and retire it,
                # letting _ensure_midi_scan_worker start a fresh one.
                self.log("MIDI device scan timed out -- retrying.")
                self.midi_scan_id += 1
                self.midi_scan_in_progress = False
                self.midi_scan_worker_generation += 1
                self._midi_scan_worker_alive = False

            if not self.midi_scan_in_progress:
                self._start_midi_scan(update_combo=False)

        self.midi_autoconnect_after_id = self.root.after(
            2000, self.poll_midi_autoconnect
        )

    def _process_midi_port_list(self, port_names):
        """Runs on the main thread with an already-fetched port list --
        decides whether to disconnect a vanished device or auto-connect a
        newly available one. Never touches mido/rtmidi itself."""
        if port_names is None:
            return

        current_ports = set(port_names)
        if current_ports != self.midi_known_ports:
            # Plugged-in devices changed -- treat as fresh physical state
            # even if the user previously clicked Disconnect.
            self.midi_known_ports = current_ports
            self.midi_auto_connect_suspended = False

            if (
                self.midi_input is not None
                and getattr(self.midi_input, "name", None) not in current_ports
            ):
                self.disconnect_midi_input(update_status=True)

            if self.midi_window is not None and self.midi_window.winfo_exists():
                self.refresh_midi_ports()

        if (
            self.midi_input is None
            and not getattr(self, "midi_open_in_progress", False)
            and not self.midi_auto_connect_suspended
            and port_names
        ):
            self._connect_midi_port(port_names[0])

    def handle_midi_message(self, message):
        # RtMidi invokes this outside Tk's main thread -- never touch Tk
        # widgets here; root.after(0, ...) can deadlock on shutdown. Queue
        # put is safe; poll_midi_note_queue drains it on the main thread.
        try:
            message_copy = message.copy()
        except Exception:
            message_copy = message
        self.midi_note_queue.put(message_copy)

    def poll_midi_note_queue(self):
        try:
            while True:
                message = self.midi_note_queue.get_nowait()
                self.process_midi_message(message)
        except queue.Empty:
            pass

        # Fast poll -- live playing needs low latency, unlike scan/connect
        # results on the slower midi_queue poll.
        self.midi_note_queue_after_id = self.root.after(5, self.poll_midi_note_queue)

    def process_midi_message(self, message):
        if message.type == "note_on" and message.velocity > 0:
            raw_note = int(message.note)
            velocity = int(message.velocity)
            # Transpose so physical Middle C always plays the center of
            # whatever instrument's range is currently loaded, not raw
            # note 60 -- recomputed from self.home_base_midi every note_on.
            played_note = raw_note + (self.home_base_midi - 60)
            self.midi_note_transposition[raw_note] = played_note
            self.midi_held_notes.add(played_note)
            self.note_on(played_note, velocity=velocity, source="midi")
            self._update_octave_label_for_midi_note(played_note)
            return

        if message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            raw_note = int(message.note)
            # Release whatever note_on actually played for this key, not a
            # freshly recomputed transposition -- could differ if a
            # different instrument loaded while the key was still held.
            played_note = self.midi_note_transposition.pop(raw_note, raw_note)
            self.midi_held_notes.discard(played_note)
            self.note_off(played_note, source="midi")
            return

        # Standard MIDI Volume controller (CC7): drive the VOLUME fader.
        if message.type == "control_change" and message.control == 7:
            volume_value = int(message.value) / 127.0 * 100.0
            if "VOLUME" in self.controls:
                self.controls["VOLUME"].set(volume_value)
            self.on_control_change("VOLUME", volume_value)
            return

        # Pitch bend wheel: mido reports message.pitch in -8192..8191.
        if message.type == "pitchwheel":
            normalized = max(-1.0, min(1.0, message.pitch / 8192.0))
            if self.pitch_wheel is not None:
                self.pitch_wheel.set(normalized)
            self.on_pitch_wheel_change(normalized)
            return

        # Standard MIDI Mod Wheel (CC1): drives vibrato depth.
        if message.type == "control_change" and message.control == 1:
            raw_cc = int(message.value)
            if (
                self._last_mod_wheel_cc is not None
                and abs(raw_cc - self._last_mod_wheel_cc) < 2
            ):
                return
            self._last_mod_wheel_cc = raw_cc

            normalized = raw_cc / 127.0
            if self.mod_wheel_widget is not None:
                self.mod_wheel_widget.set(normalized)
            self.on_mod_wheel_change(normalized)
            return

        # Standard MIDI All Sound Off / All Notes Off messages.
        if message.type == "control_change" and message.control in {120, 123}:
            self.release_all_midi_notes()
            return

        # Nothing above matched -- log it so an unmapped control (e.g. an
        # octave up/down button, which has no standard MIDI message) can be
        # identified from the log and then wired up.
        self.log(f"Unhandled MIDI message: {message}")

    def release_all_midi_notes(self):
        notes = self.midi_held_notes
        self.midi_held_notes = set()
        self.midi_note_transposition = {}

        for note in list(notes):
            if self.engine is not None:
                self.engine.note_off(note)
            self.release_key_visual(note)

    def on_drop_enter(self, event):
        self.drop_zone.config(bg="#40475a", text="Release to use this WAV file")
        return event.action

    def on_drop_leave(self, event):
        self.update_drop_zone()
        return event.action

    def parse_dropped_paths(self, raw_data):
        return [Path(item) for item in self.root.tk.splitlist(raw_data)]

    def _drop_zone_default_text(self):
        # Generated from SUPPORTED_AUDIO_EXTENSIONS (the set actually
        # validated in _adopt_input_audio_source) instead of a hand-typed
        # list, so this can't silently drift out of sync with what the
        # app really accepts.
        extensions = " · ".join(
            ext.lstrip(".").upper() for ext in sorted(SUPPORTED_AUDIO_EXTENSIONS)
        )
        return (
            f"Drag & drop audio here\n{extensions}\n"
            f"Current: {self.selected_input_name}"
        )

    def update_drop_zone(self, message=None, error=False):
        if message is None:
            message = self._drop_zone_default_text()
        self.drop_zone.config(
            text=message,
            bg="#2b2e38",
            fg="#ff8a80" if error else "#eeeeee",
        )

    def find_ffmpeg(self):
        """Return an ffmpeg executable path if available."""
        candidates = [
            shutil.which("ffmpeg"),
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]

        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate

        return None

    def convert_audio_to_target_wav(self, source: Path):
        """
        Convert common audio/video formats into the backend's fixed input file:
        input/target.wav

        For video files such as .mp4, only the first audio stream is used.
        """
        source = Path(source)

        if source.suffix.lower() == ".wav":
            if source.resolve() != INPUT_AUDIO.resolve():
                shutil.copy2(source, INPUT_AUDIO)
            return

        ffmpeg = self.find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg is required for MP3/M4A/MP4/FLAC conversion.\n\n"
                "Install it on macOS with:\n"
                "brew install ffmpeg"
            )

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(INPUT_AUDIO),
        ]

        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            error_text = result.stderr.strip()
            if len(error_text) > 1200:
                error_text = error_text[-1200:]
            raise RuntimeError("Audio conversion failed.\n\n" f"{error_text}")

    def on_audio_drop(self, event):
        try:
            paths = self.parse_dropped_paths(event.data)

            if len(paths) != 1:
                self.update_drop_zone(
                    "Please drop exactly one audio file.",
                    error=True,
                )
                return event.action

            self._adopt_input_audio_source(paths[0])

        except Exception as exc:
            self.update_drop_zone(
                f"Could not load audio:\n{exc}",
                error=True,
            )
            self.log(f"Drop error: {exc}")
            messagebox.showerror(
                "Audio import failed",
                str(exc),
            )

        return event.action

    def browse_for_audio_file(self, event=None):
        supported = sorted(SUPPORTED_AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[
                ("Audio files", " ".join(f"*{ext}" for ext in supported)),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self._adopt_input_audio_source(Path(path))
        except Exception as exc:
            self.update_drop_zone(
                f"Could not load audio:\n{exc}",
                error=True,
            )
            self.log(f"Browse error: {exc}")
            messagebox.showerror(
                "Audio import failed",
                str(exc),
            )

    def _adopt_input_audio_source(self, source):
        """Validate, convert, and adopt `source` as the current input
        audio. Shared by drag-and-drop and the file browser."""
        if self.run_in_progress:
            # Run System reads INPUT_AUDIO from a background subprocess --
            # loading new audio here would overwrite it mid-run.
            raise ValueError(
                "Wait for the current run to finish before loading new audio."
            )

        if not source.is_file():
            raise ValueError("The selected item is not a file.")

        extension = source.suffix.lower()

        if extension not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(f"Unsupported format: {extension}\nSupported: {supported}")

        INPUT_AUDIO.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.update_drop_zone("Preparing audio...\n" f"{source.name}")
        self.root.update_idletasks()

        self.convert_audio_to_target_wav(source)

        self.selected_input_name = source.name
        self.audio_uploaded = True
        self.reset_run_state()

        self.update_drop_zone("Audio ready\n" f"{source.name}\n" "Press Run System")

        self.log(f"New source selected: {source}")
        self.log(f"Converted/copied to backend input: {INPUT_AUDIO}")

    def build_knob_panel(self, parent, title, names, col):
        box = self._bordered_panel(parent, title)
        box.grid(row=0, column=col, padx=5, pady=2, sticky="nsew")

        content = tk.Frame(box, bg="#202120")
        content.pack(fill="both", expand=True)
        content.grid_rowconfigure(0, weight=1)
        for i in range(len(names)):
            content.grid_columnconfigure(i, weight=1, uniform=f"{title}_faders")

        short_labels = {
            "EQ_FREQ": "FREQ",
            "EQ_BOOST": "GAIN",
            "ATTACK": "ATTK",
            "DECAY": "DCY",
            "SUSTAIN": "SUS",
            "RELEASE": "REL",
        }
        for i, name in enumerate(names):
            frame = tk.Frame(content, bg="#202120", bd=0)
            frame.grid(row=0, column=i, sticky="nsew")
            if i > 0:
                tk.Frame(frame, bg="#0c0c0c", width=1).pack(side="left", fill="y")
            if name == "EQ_FREQ":
                min_value, max_value, resolution = 0, 100, 0.1
            elif name == "EQ_BOOST":
                min_value, max_value, resolution = -12, 12, 0.5
            else:
                min_value, max_value, resolution = 0, 100, 1

            fader = MixerFader(
                frame,
                label=short_labels.get(name, name),
                initial=DEFAULT_CONTROLS.get(name, 50),
                min_value=min_value,
                max_value=max_value,
                resolution=resolution,
                command=lambda value, n=name: self.on_control_change(n, value),
                value_formatter=lambda value, n=name: self.format_control_value(
                    n, value
                ),
                scale_marks=control_scale_marks(name),
                width=118,
                height=470,
            )
            fader.pack(side="top", expand=True, fill="both")
            self.controls[name] = fader

    def build_output_panel(self, parent, col=4):
        box = self._bordered_panel(parent, "OUTPUT")
        box.grid(row=0, column=col, padx=(8, 0), pady=2, sticky="nsew")

        top_row = tk.Frame(box, bg="#202120")
        top_row.pack(fill="x", padx=5, pady=(2, 0))

        volume_fader = MixerFader(
            top_row,
            label="VOL",
            initial=DEFAULT_CONTROLS["VOLUME"],
            min_value=0,
            max_value=100,
            resolution=1,
            command=lambda value: self.on_control_change("VOLUME", value),
            value_formatter=lambda value: self.format_control_value("VOLUME", value),
            scale_marks=control_scale_marks("VOLUME"),
            width=116,
            height=365,
        )
        volume_fader.pack(side="left", fill="y")
        self.controls["VOLUME"] = volume_fader

        meters_frame = tk.Frame(top_row, width=54, height=350, bg="#202120")
        meters_frame.pack(side="left", padx=(3, 0), pady=(10, 0))
        meters_frame.pack_propagate(False)
        self.output_meter = self.create_level_meter(
            meters_frame, title="OUTPUT", column=0
        )
        self.clip_label = tk.Label(
            meters_frame,
            text="CLIP",
            width=10,
            fg="#555555",
            bg="#202120",
            font=(self.ui_font, 9, "bold"),
            anchor="center",
        )
        self.clip_label.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        self.clip_label.bind("<Button-1>", lambda event: self.clear_clip_indicator())

        def small_button(text, command):
            button = self._styled_button(box, text, command)
            button.pack(fill="x", padx=12, pady=2, ipady=2)
            return button

        self._section_divider(box, "PLAYBACK")
        self.play_input_button = small_button(
            "Play Input", lambda: self.play_one_shot(INPUT_AUDIO)
        )
        self.play_output_button = small_button(
            "Play Output", lambda: self.play_one_shot(RECONSTRUCTED_AUDIO)
        )

        self._section_divider(box, "DEFAULTS")
        small_button("Reset All", self.reset_timbre_default)

        self.start_meter_update()

    def create_level_meter(self, parent, title, column):
        frame = tk.Frame(
            parent,
            width=50,
            height=262,
            bg="#161616",
        )
        frame.grid(row=0, column=column, sticky="n", padx=1)
        frame.grid_propagate(False)

        title_label = tk.Label(
            frame,
            text=title,
            width=7,
            fg="#eeeeee",
            bg="#161616",
            font=("Arial", 8, "bold"),
            anchor="center",
        )
        title_label.pack(pady=(2, 4))

        canvas = tk.Canvas(
            frame,
            width=46,
            height=205,
            bg="#080808",
            highlightthickness=1,
            highlightbackground="#666666",
        )
        canvas.pack()

        # Meter bar background.
        bar_x0 = 9
        bar_x1 = 25
        top = 8
        bottom = 191

        canvas.create_rectangle(
            bar_x0,
            top,
            bar_x1,
            bottom,
            fill="#151515",
            outline="",
        )

        fill = canvas.create_rectangle(
            bar_x0,
            bottom,
            bar_x1,
            bottom,
            fill="#39d353",
            outline="",
        )

        peak_line = canvas.create_line(
            bar_x0 - 1,
            bottom,
            bar_x1 + 1,
            bottom,
            fill="#ffffff",
            width=2,
        )

        # Clear dB scale.
        # Major marks have labels; minor marks are shorter.
        major_ticks = [0, -6, -12, -18, -24, -36, -48, -60]
        minor_ticks = [-3, -9, -15, -21, -30, -42, -54]

        for db in minor_ticks:
            y = self.db_to_meter_y(db)
            canvas.create_line(
                bar_x1 + 2,
                y,
                bar_x1 + 6,
                y,
                fill="#555555",
                width=1,
            )

        for db in major_ticks:
            y = self.db_to_meter_y(db)
            canvas.create_line(
                bar_x1 + 2,
                y,
                bar_x1 + 8,
                y,
                fill="#888888",
                width=1,
            )
            label = "0" if db == 0 else str(db)
            canvas.create_text(
                37,
                y,
                text=label,
                fill="#bcbcbc",
                font=("Arial", 6),
                anchor="center",
            )

        value_label = tk.Label(
            frame,
            text="-∞ dB",
            width=8,
            fg="#eeeeee",
            bg="#161616",
            font=("Arial", 8, "bold"),
            anchor="center",
        )
        value_label.pack(pady=(4, 0))

        return {
            "canvas": canvas,
            "fill": fill,
            "peak_line": peak_line,
            "value_label": value_label,
            "bar_x0": bar_x0,
            "bar_x1": bar_x1,
            "bottom": bottom,
        }

    def amplitude_to_db(self, amplitude):
        amplitude = max(float(amplitude), 1e-8)
        return 20.0 * np.log10(amplitude)

    def db_to_meter_y(self, db):
        db = max(-60.0, min(0.0, float(db)))
        top = 8
        bottom = 191
        ratio = (db + 60.0) / 60.0
        return bottom - ratio * (bottom - top)

    def clear_clip_indicator(self):
        if self.engine is not None:
            with self.engine.lock:
                self.engine.clip_detected = False

        if self.clip_label is not None:
            self.clip_label.config(fg="#555555")

    def start_meter_update(self):
        if self.meter_after_id is None:
            self.update_output_meter()

    def update_single_meter(self, meter, peak):
        if not meter:
            return

        db = self.amplitude_to_db(peak)
        y = self.db_to_meter_y(db)

        if db >= -2.0:
            color = "#ff4040"
        elif db >= -6.0:
            color = "#ffb000"
        else:
            color = "#39d353"

        canvas = meter["canvas"]
        x0 = meter["bar_x0"]
        x1 = meter["bar_x1"]
        bottom = meter["bottom"]

        canvas.coords(meter["fill"], x0, y, x1, bottom)
        canvas.itemconfig(meter["fill"], fill=color)
        canvas.coords(meter["peak_line"], x0 - 1, y, x1 + 1, y)

        text = "-∞ dB" if peak <= 1e-7 else f"{db:.1f} dB"
        meter["value_label"].config(text=text)

    def update_output_meter(self):
        output_peak = 0.0
        clipped = False

        if self.engine is not None:
            levels = self.engine.get_meter_levels()
            output_peak = levels["output_level_peak"]
            clipped = levels["clip_detected"]

            with self.engine.lock:
                self.engine.input_level_peak *= 0.82
                self.engine.output_level_peak *= 0.82

        self.update_single_meter(self.output_meter, output_peak)

        if self.clip_label is not None:
            self.clip_label.config(
                fg="#ff4040" if clipped else "#555555",
            )

        self.meter_after_id = self.root.after(
            45,
            self.update_output_meter,
        )

    def bind_computer_keyboard(self):
        self.root.bind("<KeyPress>", self.on_keyboard_press)
        self.root.bind("<KeyRelease>", self.on_keyboard_release)
        self.root.focus_set()

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def format_control_value(self, name, value):
        value = float(value)

        if name == "EQ_FREQ":
            frequency = normalized_to_log_frequency(value)
            if frequency >= 1000:
                return f"{frequency / 1000:.2f} kHz"
            return f"{frequency:.0f} Hz"

        if name == "EQ_BOOST":
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.1f} dB"

        if name == "VOLUME":
            return f"{value:.0f}%"

        if name == "ATTACK":
            return f"{map_attack(value):.3f} s"

        if name == "DECAY":
            return f"{map_decay(value):.2f} s"

        if name == "SUSTAIN":
            return f"{value:.0f}%"

        if name == "RELEASE":
            return f"{map_release(value):.2f} s"

        return f"{value:.0f}"

    def get_control_values(self):
        values = {}
        for name, scale in self.controls.items():
            values[name] = float(scale.get())
        return values

    def apply_controls_to_engine(self):
        if self.engine is None:
            return

        values = self.get_control_values()

        # Extra 0.90 keeps headroom when EQ/FX/polyphony stack.
        volume = (
            volume_percent_to_gain(
                values.get(
                    "VOLUME",
                    DEFAULT_CONTROLS["VOLUME"],
                )
            )
            * 0.90
        )

        eq_freq_hz = normalized_to_log_frequency(
            values.get(
                "EQ_FREQ",
                DEFAULT_CONTROLS["EQ_FREQ"],
            )
        )

        eq_gain_db = values.get(
            "EQ_BOOST",
            DEFAULT_CONTROLS["EQ_BOOST"],
        )

        # Real-time demo safety: avoid violent peaks from extreme boosts.
        eq_gain_db = max(-10.0, min(10.0, eq_gain_db))

        # No Reverb/Delay controls on the panel; keep the effect off.
        reverb_mix = 0.0
        delay_mix = 0.0

        attack_seconds = map_attack(
            values.get(
                "ATTACK",
                DEFAULT_CONTROLS["ATTACK"],
            )
        )
        decay_seconds = map_decay(
            values.get(
                "DECAY",
                DEFAULT_CONTROLS["DECAY"],
            )
        )
        sustain_level = (
            values.get(
                "SUSTAIN",
                DEFAULT_CONTROLS["SUSTAIN"],
            )
            / 100.0
        )
        release_seconds = map_release(
            values.get(
                "RELEASE",
                DEFAULT_CONTROLS["RELEASE"],
            )
        )

        self.engine.set_controls(
            volume=volume,
            eq_freq_hz=eq_freq_hz,
            eq_gain_db=eq_gain_db,
            reverb_mix=reverb_mix,
            delay_mix=delay_mix,
            attack_seconds=attack_seconds,
            decay_seconds=decay_seconds,
            sustain_level=sustain_level,
            release_seconds=release_seconds,
        )

    def on_control_change(self, name, value):
        self.apply_controls_to_engine()

    def on_pitch_wheel_change(self, normalized_value):
        """normalized_value is -1..+1 (the pitch wheel's own range,
        matching mido's pitchwheel convention) -- scaled to semitones only
        here, at the boundary into the audio engine. Up and down use
        independent ranges (from the NumberStepper pair next to the
        wheel), so a full-up deflection and a full-down deflection can
        bend by different amounts."""
        if self.engine is not None:
            if normalized_value >= 0:
                semitones = normalized_value * self.pitch_bend_range_up_semitones
            else:
                semitones = -normalized_value * self.pitch_bend_range_down_semitones
            self.engine.set_controls(pitch_bend_semitones=semitones)

    def on_mod_wheel_change(self, normalized_value):
        if self.engine is not None:
            self.engine.set_controls(mod_wheel=normalized_value)

    def _default_base_midi_for_range(self):
        """The 'home' base note (key 'a', offset 0) for the current 36
        rendered notes -- the C nearest the middle of [min_midi, max_midi],
        so the playable window starts near the middle of the instrument's
        range instead of always pinned toward the bottom.
        """
        if not self.wav_by_midi or self.min_midi is None or self.max_midi is None:
            return 60

        candidate_bases = [
            midi_note
            for midi_note in sorted(self.wav_by_midi.keys())
            if midi_note % 12 == 0
        ]
        if not candidate_bases:
            return self.min_midi

        midpoint = (self.min_midi + self.max_midi) / 2
        return min(candidate_bases, key=lambda midi_note: abs(midi_note - midpoint))

    def reset_timbre_default(self):
        for name, value in DEFAULT_CONTROLS.items():
            if name in self.controls:
                self.controls[name].set(value)

        # Reset octave base too.
        if self.min_midi is not None and self.max_midi is not None:
            self.base_midi = self._default_base_midi_for_range()
            self.home_base_midi = self.base_midi
            self.octave_shift = 0
            self.update_octave_label()

        self.apply_controls_to_engine()
        if self.engine is not None:
            self.engine.reset_fx_buffers()
        self.log("Reset to default dry/original timbre settings.")

    def save_timbre_from_main(self):
        """Open Timbre Manager and immediately start saving the current learned timbre."""
        if not LEARNED_PARAMS.exists():
            messagebox.showerror(
                "No learned timbre",
                "Run System first, then press Save Timbre.",
            )
            return
        if self.run_in_progress:
            messagebox.showerror(
                "System busy",
                "Wait for the current run to finish before saving -- it's "
                "still writing the learned timbre files.",
            )
            return

        self.open_sound_manager()
        # Let the new Toplevel register with the window server before
        # opening a modal on top of it (see ask_string_dialog's docstring).
        self.root.after(400, self.manager_save_current_sound)

    def play_one_shot(self, path: Path):
        """Play a WAV preview with DC removal, headroom, and anti-pop fades."""
        if not path.exists():
            messagebox.showerror("Missing file", str(path))
            return

        if sd is None:
            messagebox.showerror(
                "Audio unavailable",
                "sounddevice is not installed. Run: python -m pip install sounddevice",
            )
            return

        generation = self.preview_generation

        def worker():
            try:
                audio, sample_rate = sf.read(path, dtype="float32")

                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                audio = np.asarray(audio, dtype=np.float32)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

                if len(audio) == 0:
                    raise RuntimeError(f"Audio file is empty: {path}")

                # Remove DC offset. A non-zero waveform center can cause a pop
                # when playback starts or stops.
                audio = audio - np.mean(audio)

                # Keep preview playback below full scale so the source itself
                # cannot clip the output device.
                peak = float(np.max(np.abs(audio)) + 1e-8)
                if peak > 0.80:
                    audio = audio / peak * 0.80

                # Always begin and end at zero. This prevents the sudden waveform
                # discontinuity that produces the audible "pop" sound.
                fade_in_samples = min(len(audio) // 2, max(1, int(sample_rate * 0.015)))
                fade_out_samples = min(
                    len(audio) // 2, max(1, int(sample_rate * 0.030))
                )

                if fade_in_samples > 1:
                    fade_in = (
                        np.sin(
                            np.linspace(
                                0.0, np.pi / 2.0, fade_in_samples, dtype=np.float32
                            )
                        )
                        ** 2
                    )
                    audio[:fade_in_samples] *= fade_in

                if fade_out_samples > 1:
                    fade_out = (
                        np.cos(
                            np.linspace(
                                0.0, np.pi / 2.0, fade_out_samples, dtype=np.float32
                            )
                        )
                        ** 2
                    )
                    audio[-fade_out_samples:] *= fade_out

                # Tiny silence guard so the audio device settles before/after.
                guard = np.zeros(max(1, int(sample_rate * 0.010)), dtype=np.float32)
                audio = np.concatenate((guard, audio, guard))

                if self.preview_generation != generation:
                    # Something else (e.g. Load) claimed audio priority
                    # while this was decoding -- skip playback.
                    return

                # Fades an outgoing clip before switching, instead of
                # sd.play()/sd.stop()'s abrupt (poppy) cut.
                self.preview_player.play(audio, sample_rate)
                self.run_queue.put(("oneshot_log", f"Playing: {path.name}"))
            except Exception as exc:
                self.run_queue.put(("oneshot_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self.poll_auxiliary_queue)

    def poll_auxiliary_queue(self, attempts_remaining=100):
        try:
            while True:
                message_type, payload = self.run_queue.get_nowait()

                if message_type == "oneshot_log":
                    self.log(payload)
                elif message_type == "oneshot_error":
                    self.log(f"Playback error: {payload}")
                    messagebox.showerror("Playback failed", payload)
                else:
                    # Put backend-run messages back so poll_run_queue handles them.
                    self.run_queue.put((message_type, payload))
                    return
        except queue.Empty:
            # Worker hasn't finished yet -- keep checking instead of giving
            # up after one look and dropping a slower result.
            if attempts_remaining > 0:
                self.root.after(50, self.poll_auxiliary_queue, attempts_remaining - 1)

    def _selected_manager_taxonomy_row(self):
        if self.sound_manager_taxonomy_list is None:
            return None
        selection = self.sound_manager_taxonomy_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index >= len(self.sound_manager_taxonomy_rows):
            return None
        return self.sound_manager_taxonomy_rows[index]

    def _selected_manager_sound(self):
        if self.sound_manager_sound_list is None:
            return None
        selection = self.sound_manager_sound_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index >= len(self.sound_manager_sound_records):
            return None
        return self.sound_manager_sound_records[index]

    def _refresh_sound_manager_to_loaded_timbre(self):
        """Navigate to whichever timbre is currently loaded in the main
        player, if any, instead of leaving the tree on its previous/first
        selection."""
        if self.loaded_timbre_location is not None:
            family_folder, instrument_folder, sound_folder = self.loaded_timbre_location
            self.refresh_sound_manager(
                select_family_folder=family_folder,
                select_instrument_folder=instrument_folder,
                select_sound_folder=sound_folder,
            )
        else:
            self.refresh_sound_manager()

    def open_sound_manager(self):
        if (
            self.sound_manager_window is not None
            and self.sound_manager_window.winfo_exists()
        ):
            self.sound_manager_window.deiconify()
            self.sound_manager_window.lift()
            self.sound_manager_window.focus_force()
            self._refresh_sound_manager_to_loaded_timbre()
            return

        window = tk.Toplevel(self.root)
        self.sound_manager_window = window
        window.title("Timbre Manager")
        window.geometry("980x610")
        window.minsize(820, 520)
        window.configure(bg="#151615")
        window.transient(self.root)
        self._center_toplevel(window)

        title = tk.Label(
            window,
            text="Timbre Manager",
            fg="#f4f4f4",
            bg="#171817",
            font=(self.ui_font, 22, "bold"),
        )
        title.pack(fill="x", pady=(0, 10), ipady=10)

        subtitle = tk.Label(
            window,
            text=(
                "Timbres are filed automatically by family and instrument, "
                "the same layout the classifier's training data uses."
            ),
            fg="#aeb2b5",
            bg="#151615",
            font=(self.ui_font, 10),
        )
        subtitle.pack(fill="x", padx=18, pady=(0, 10))

        body = tk.Frame(window, bg="#151615")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=2)

        category_panel = self._bordered_panel(body, "FAMILY / INSTRUMENT")
        category_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        category_scroll = tk.Scrollbar(category_panel)
        category_scroll.pack(side="right", fill="y", padx=(0, 8), pady=10)
        self.sound_manager_taxonomy_list = tk.Listbox(
            category_panel,
            bg="#0d0e0d",
            fg="#eeeeee",
            selectbackground="#416b88",
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            exportselection=False,
            font=(self.ui_font, 11),
            yscrollcommand=category_scroll.set,
            # Without an explicit width, a Listbox's requested size tracks
            # its longest row despite pack(fill="both") stretching its
            # display -- caused visible window resize on row changes.
            width=1,
        )
        self.sound_manager_taxonomy_list.pack(
            fill="both", expand=True, padx=(10, 0), pady=10
        )
        category_scroll.config(command=self.sound_manager_taxonomy_list.yview)
        self.sound_manager_taxonomy_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self.on_manager_taxonomy_row_selected(),
        )

        sound_panel = self._bordered_panel(body, "TIMBRES")
        sound_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        sound_scroll = tk.Scrollbar(sound_panel)
        sound_scroll.pack(side="right", fill="y", padx=(0, 8), pady=(10, 118))
        self.sound_manager_sound_list = tk.Listbox(
            sound_panel,
            bg="#0d0e0d",
            fg="#eeeeee",
            selectbackground="#416b88",
            selectforeground="#ffffff",
            relief="flat",
            bd=0,
            exportselection=False,
            font=(self.ui_font, 11),
            yscrollcommand=sound_scroll.set,
            # See sound_manager_taxonomy_list -- pin the requested width so
            # switching to a category with a longer/shorter/empty sound
            # list can't shift the whole window's natural size.
            width=1,
        )
        self.sound_manager_sound_list.pack(
            fill="both", expand=True, padx=(10, 0), pady=(10, 0)
        )
        sound_scroll.config(command=self.sound_manager_sound_list.yview)
        self.sound_manager_sound_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self.update_manager_sound_details(),
        )
        self.sound_manager_sound_list.bind(
            "<Double-Button-1>",
            lambda _event: self.manager_load_sound(),
        )
        self.sound_manager_sound_list.bind(
            "<Return>",
            lambda _event: self.manager_load_sound(),
        )
        self.sound_manager_sound_list.bind(
            "<space>",
            lambda _event: self.manager_preview_sound(),
        )
        # macOS reports right-click as Button-2 or Button-3 depending on
        # input device, plus Control-click for trackpad users.
        for sequence in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            self.sound_manager_sound_list.bind(
                sequence, self._show_sound_manager_context_menu
            )

        self.sound_manager_details_label = tk.Label(
            sound_panel,
            text="Select a timbre to view details.",
            fg="#aeb2b5",
            bg="#202120",
            justify="left",
            anchor="w",
            font=(self.ui_font, 9),
            # width pins a constant minimum so the window doesn't visibly
            # resize as this text swings between long and short content;
            # wraplength is a safety net for unusually long timbre names.
            width=95,
            wraplength=560,
        )
        self.sound_manager_details_label.pack(fill="x", padx=10, pady=(8, 5))

        sound_buttons_1 = tk.Frame(sound_panel, bg="#202120")
        sound_buttons_1.pack(fill="x", padx=8, pady=2)
        self.save_sound_button = self._manager_button(
            sound_buttons_1, "Save Current Timbre", self.manager_save_current_sound
        )
        self.save_sound_button.pack(side="left", expand=True, fill="x", padx=2)
        self._set_button_enabled(self.save_sound_button, self.timbre_ready)
        self._manager_button(sound_buttons_1, "Load", self.manager_load_sound).pack(
            side="left", expand=True, fill="x", padx=2
        )
        self._manager_button(
            sound_buttons_1, "▶ Play Preview", self.manager_preview_sound
        ).pack(side="left", expand=True, fill="x", padx=2)

        sound_buttons_2 = tk.Frame(sound_panel, bg="#202120")
        sound_buttons_2.pack(fill="x", padx=8, pady=2)
        self._manager_button(sound_buttons_2, "Rename", self.manager_rename_sound).pack(
            side="left", expand=True, fill="x", padx=2
        )
        self._manager_button(sound_buttons_2, "Delete", self.manager_delete_sound).pack(
            side="left", expand=True, fill="x", padx=2
        )

        tk.Label(
            sound_panel,
            text="CNN CLASSIFIER (also refiles the timbre to match)",
            fg="#f8d36a",
            bg="#202120",
            font=(self.ui_font, 8, "bold"),
        ).pack(fill="x", padx=8, pady=(4, 0))

        sound_buttons_3 = tk.Frame(sound_panel, bg="#202120")
        sound_buttons_3.pack(fill="x", padx=8, pady=(2, 10))
        self._cnn_button(
            sound_buttons_3, "Correct CNN Label", self.manager_fix_classification
        ).pack(side="left", expand=True, fill="x", padx=2)
        self._cnn_button(
            sound_buttons_3, "Retrain Classifier", self.retrain_classifier_threaded
        ).pack(side="left", expand=True, fill="x", padx=2)

        window.protocol("WM_DELETE_WINDOW", self.close_sound_manager)
        self._refresh_sound_manager_to_loaded_timbre()

    def _manager_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=(self.ui_font, 9),
            bg="#d7d7d7",
            fg="#111111",
            # Without this, Tk falls back to its own washed-out default
            # disabled-text color, which reads as nearly invisible on
            # macOS's native (always-light) button face.
            disabledforeground="#6b6b6b",
            activebackground="#eeeeee",
            activeforeground="#000000",
            relief="raised",
            bd=1,
            cursor="hand2",
            pady=4,
        )

    def _cnn_button(self, parent, text, command):
        # macOS Tk buttons mostly ignore custom bg/fg on the native button
        # face -- only the highlight border reliably shows through -- so
        # differentiate these with a colored ring instead of a fill color.
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=(self.ui_font, 9, "bold"),
            fg="#7a5200",
            disabledforeground="#8a7c5e",
            activeforeground="#5c3d00",
            relief="raised",
            bd=1,
            cursor="hand2",
            pady=4,
            highlightbackground="#f8d36a",
            highlightcolor="#f8d36a",
            highlightthickness=2,
        )

    def _danger_button(self, parent, text, command):
        # Same highlight-ring approach as _cnn_button (macOS Tk ignores a
        # custom fill on the native button face) -- red ring instead of
        # gold, for a genuinely destructive/irreversible action.
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=(self.ui_font, 9, "bold"),
            fg="#8a2a20",
            disabledforeground="#8a7c5e",
            activeforeground="#5c1a12",
            relief="raised",
            bd=1,
            cursor="hand2",
            pady=4,
            highlightbackground="#e05a4e",
            highlightcolor="#e05a4e",
            highlightthickness=2,
        )

    def _count_correction_samples(self, instrument_class):
        """
        Number of correction_*.wav files filed for this instrument under
        data/<family>/<instrument>/ -- deliberately only the ones added via
        the correction flow (add_correction_to_training_data's naming
        pattern), not the original baseline dataset, so the retrain-prompt
        threshold tracks corrections actually added, not an arbitrary
        round total the instrument already happened to have.
        """
        try:
            class_dir = DATA_DIR / family_of(instrument_class) / instrument_class
        except ValueError:
            return 0
        if not class_dir.exists():
            return 0
        return sum(1 for _ in class_dir.glob("correction_*.wav"))

    def _maybe_prompt_retrain(self, instrument_class, parent):
        """
        Returns True if the user chose to retrain right now (caller should
        skip its own success message/dialog and hand off to do_retrain),
        False otherwise -- either the threshold wasn't hit, or it was and
        the user declined.
        """
        count = self._count_correction_samples(instrument_class)
        if count == 0 or count % RETRAIN_PROMPT_EVERY != 0:
            return False
        return messagebox.askyesno(
            "Retrain classifier?",
            f"You've now added {count} corrected samples for "
            f"{taxonomy_display_name(instrument_class)}. Retrain the "
            "classifier now?",
            parent=parent,
        )

    def _maybe_prompt_new_instrument_retrain(self, instrument_class, parent):
        """
        The first time a brand-new (not-yet-trained) instrument's data/
        folder reaches NEW_CLASS_MIN_SAMPLES, offer to retrain right then
        -- a different, more meaningful moment than
        _maybe_prompt_retrain's periodic every-10 prompt for an
        already-known class. Fires at most once per instrument, since the
        exact-equality check only matches on the call where the count
        first reaches this threshold.
        """
        trained_classes = set(self._load_class_names())
        if instrument_class in trained_classes:
            return False

        try:
            folder = DATA_DIR / family_of(instrument_class) / instrument_class
        except ValueError:
            return False
        if not folder.is_dir():
            return False
        count = sum(
            1
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )
        if count != NEW_CLASS_MIN_SAMPLES:
            return False

        return messagebox.askyesno(
            "Retrain classifier?",
            f"{taxonomy_display_name(instrument_class)} now has "
            f"{NEW_CLASS_MIN_SAMPLES} training samples -- likely enough "
            "to be recognized. Retrain the classifier now?",
            parent=parent,
        )

    def _load_class_names(self):
        """The trained classifier's class list from models/class_names.json,
        or [] if it's missing/unreadable (not yet trained)."""
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _pending_new_instruments(self):
        """
        Instruments registered in the taxonomy but not yet part of the
        currently trained classifier (models/class_names.json), with how
        many training-audio samples have been collected for each so far
        under data/<family>/<instrument>/.
        """
        trained_classes = set(self._load_class_names())

        pending = []
        for name in INSTRUMENT_NAMES:
            if name in trained_classes:
                continue
            folder = DATA_DIR / family_of(name) / name
            count = 0
            if folder.is_dir():
                count = sum(
                    1
                    for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                )
            pending.append({"name": name, "count": count})
        return pending

    def _data_counts_by_family(self):
        """
        Training-audio file count under data/<family>/<instrument>/ for
        every known instrument (trained or not), grouped by family --
        same counting rule as _pending_new_instruments, just covering the
        whole taxonomy instead of only not-yet-trained instruments.
        """
        counts_by_family = {family: [] for family in FAMILY_NAMES}
        for name in INSTRUMENT_NAMES:
            family = family_of(name)
            folder = DATA_DIR / family / name
            count = 0
            if folder.is_dir():
                count = sum(
                    1
                    for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                )
            counts_by_family.setdefault(family, []).append(
                {"name": name, "count": count}
            )
        return counts_by_family

    def open_data_overview(self):
        if self.data_overview_window is not None and self.data_overview_window.winfo_exists():
            self.data_overview_window.deiconify()
            self.data_overview_window.lift()
            self.refresh_data_overview()
            return

        window = tk.Toplevel(self.root)
        self.data_overview_window = window
        window.title("Data Overview")
        window.geometry("380x520")
        window.minsize(340, 400)
        window.configure(bg="#151615")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self.close_data_overview_window)
        self._center_toplevel(window)

        title = tk.Label(
            window,
            text="Data Overview",
            fg="#f4f4f4",
            bg="#171817",
            font=(self.ui_font, 18, "bold"),
        )
        title.pack(fill="x", pady=(0, 8), ipady=10)

        subtitle = tk.Label(
            window,
            text="Training audio currently on disk, by family and instrument.",
            fg="#aeb2b5",
            bg="#151615",
            font=(self.ui_font, 9),
            wraplength=340,
            justify="left",
        )
        subtitle.pack(fill="x", padx=16, pady=(0, 8))

        self.data_overview_total_label = tk.Label(
            window,
            text="",
            fg="#f8d36a",
            bg="#151615",
            font=(self.ui_font, 13, "bold"),
        )
        self.data_overview_total_label.pack(fill="x", padx=16, pady=(0, 8))

        body = tk.Frame(window, bg="#151615")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        scroll = tk.Scrollbar(body)
        scroll.pack(side="right", fill="y")
        self.data_overview_list = tk.Listbox(
            body,
            bg="#0d0e0d",
            fg="#eeeeee",
            relief="flat",
            bd=0,
            exportselection=False,
            font=(self.ui_font, 11),
            yscrollcommand=scroll.set,
            width=1,
        )
        self.data_overview_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.data_overview_list.yview)
        self.data_overview_list.bind(
            "<MouseWheel>",
            lambda event: self.data_overview_list.yview_scroll(
                -1 if event.delta > 0 else 1, "units"
            ),
        )

        close_button = self._styled_button(window, "Close", self.close_data_overview_window)
        close_button.pack(fill="x", padx=16, pady=(0, 16), ipady=4)

        self.refresh_data_overview()

    def close_data_overview_window(self):
        if self.data_overview_window is not None:
            try:
                self.data_overview_window.destroy()
            except Exception:
                pass
        self.data_overview_window = None
        self.data_overview_list = None
        self.data_overview_total_label = None

    def refresh_data_overview(self):
        if self.data_overview_list is None:
            return
        self.data_overview_list.delete(0, tk.END)
        counts_by_family = self._data_counts_by_family()

        if self.data_overview_total_label is not None:
            grand_total = sum(
                instrument["count"]
                for instruments in counts_by_family.values()
                for instrument in instruments
            )
            self.data_overview_total_label.config(text=f"Total: {grand_total} files")

        for family in FAMILY_NAMES:
            instruments = counts_by_family.get(family, [])
            family_total = sum(i["count"] for i in instruments)
            row_index = self.data_overview_list.size()
            self.data_overview_list.insert(tk.END, f"{family.upper()}  ({family_total})")
            self.data_overview_list.itemconfig(row_index, fg="#f8d36a")
            for instrument in sorted(instruments, key=lambda i: i["name"]):
                self.data_overview_list.insert(
                    tk.END,
                    f"      {taxonomy_display_name(instrument['name'])}"
                    f"  ({instrument['count']})",
                )

    def _load_recognition_agreement(self):
        """
        Reads models/recognition_agreement.json: {"instrument": {"agreed":
        N, "corrected": N}, ...}, accumulated live from real usage (see
        _record_recognition_agreement) rather than train_classifier.py's
        held-out test set. Returns {} if nothing has been recorded yet or
        the file is unreadable.
        """
        if not RECOGNITION_AGREEMENT_PATH.exists():
            return {}
        try:
            with open(RECOGNITION_AGREEMENT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        # A hand-edited entry that isn't {"agreed": N, "corrected": N}
        # (e.g. accidentally set to a bare number) shouldn't crash the
        # window -- drop just that entry rather than the whole file.
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    def _record_recognition_agreement(
        self, instrument_class, agreed, family_agreed=None, corrected_to=None
    ):
        """
        Tallies one real-world outcome for `instrument_class` (the CNN's
        own prediction) -- powers the Recognition Accuracy window's "when
        the CNN says X, how often is it actually right" stat, using every
        judged upload instead of a held-out test slice.

        Tracks family-level agreement separately: correcting violin to
        viola is an instrument-level miss but a family-level hit, so a
        rollup of instrument agreement would wrongly score the family 0%.
        `family_agreed` defaults to `agreed`; pass it explicitly for a
        cross-instrument correction. `corrected_to` records which
        instrument a correction went to, for display alongside the family
        score.

        Called only from manager_save_current_sound (agreed) and
        fix_current_classification (not agreed) -- not from
        manager_fix_classification, which corrects an already-saved sound
        later rather than judging this upload in the moment.
        """
        if family_agreed is None:
            family_agreed = agreed

        stats = self._load_recognition_agreement()
        entry = stats.setdefault(instrument_class, {})
        entry.setdefault("agreed", 0)
        entry.setdefault("corrected", 0)
        entry.setdefault("family_agreed", 0)
        entry.setdefault("family_corrected", 0)
        entry.setdefault("corrected_to", {})
        entry["agreed" if agreed else "corrected"] += 1
        entry["family_agreed" if family_agreed else "family_corrected"] += 1
        if corrected_to:
            entry["corrected_to"][corrected_to] = (
                entry["corrected_to"].get(corrected_to, 0) + 1
            )
        try:
            RECOGNITION_AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RECOGNITION_AGREEMENT_PATH, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except OSError as exc:
            self.log(f"Could not save recognition agreement stats: {exc}")

    def open_accuracy_overview(self):
        if self.accuracy_window is not None and self.accuracy_window.winfo_exists():
            self.accuracy_window.deiconify()
            self.accuracy_window.lift()
            self.refresh_accuracy_overview()
            return

        window = tk.Toplevel(self.root)
        self.accuracy_window = window
        window.title("Recognition Accuracy")
        window.geometry("380x520")
        window.minsize(340, 400)
        window.configure(bg="#151615")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self.close_accuracy_window)
        self._center_toplevel(window)

        title = tk.Label(
            window,
            text="Recognition Accuracy",
            fg="#f4f4f4",
            bg="#171817",
            font=(self.ui_font, 18, "bold"),
        )
        title.pack(fill="x", pady=(0, 8), ipady=10)

        subtitle = tk.Label(
            window,
            text=(
                "From real use: how often you saved a prediction as-is "
                "vs. corrected it, per instrument."
            ),
            fg="#aeb2b5",
            bg="#151615",
            font=(self.ui_font, 9),
            wraplength=340,
            justify="left",
        )
        subtitle.pack(fill="x", padx=16, pady=(0, 8))

        self.accuracy_overall_label = tk.Label(
            window,
            text="",
            fg="#f8d36a",
            bg="#151615",
            font=(self.ui_font, 13, "bold"),
        )
        self.accuracy_overall_label.pack(fill="x", padx=16, pady=(0, 8))

        body = tk.Frame(window, bg="#151615")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        scroll = tk.Scrollbar(body)
        scroll.pack(side="right", fill="y")
        self.accuracy_list = tk.Listbox(
            body,
            bg="#0d0e0d",
            fg="#eeeeee",
            relief="flat",
            bd=0,
            exportselection=False,
            font=(self.ui_font, 11),
            yscrollcommand=scroll.set,
            width=1,
        )
        self.accuracy_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.accuracy_list.yview)
        self.accuracy_list.bind(
            "<MouseWheel>",
            lambda event: self.accuracy_list.yview_scroll(
                -1 if event.delta > 0 else 1, "units"
            ),
        )
        # Same right-click coverage as _show_sound_manager_context_menu --
        # macOS reports right-click as Button-2 or Button-3 depending on
        # input device, plus Control-click for trackpad users.
        for sequence in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            self.accuracy_list.bind(sequence, self._show_accuracy_context_menu)

        bottom_buttons = tk.Frame(window, bg="#151615")
        bottom_buttons.pack(fill="x", padx=16, pady=(0, 16))
        clear_button = self._styled_button(
            bottom_buttons, "Clear History", self.clear_recognition_agreement
        )
        clear_button.pack(side="left", expand=True, fill="x", padx=(0, 4), ipady=4)
        close_button = self._styled_button(
            bottom_buttons, "Close", self.close_accuracy_window
        )
        close_button.pack(side="left", expand=True, fill="x", padx=(4, 0), ipady=4)

        self.refresh_accuracy_overview()

    def clear_recognition_agreement(self):
        """Wipes recognition_agreement.json -- the per-instrument
        agreed/corrected tallies this window is built from -- so a user can
        start the "real use" accuracy count over from zero, e.g. after a
        batch of test/throwaway judgments they don't want skewing it."""
        if not RECOGNITION_AGREEMENT_PATH.exists():
            return
        if not messagebox.askyesno(
            "Clear Judging History",
            "Permanently clear all recorded recognition accuracy history? "
            "This cannot be undone.",
            parent=self.accuracy_window,
        ):
            return
        try:
            RECOGNITION_AGREEMENT_PATH.unlink()
            self.log("Cleared recognition accuracy judging history.")
        except OSError as exc:
            messagebox.showerror("Could not clear history", str(exc))
            return
        self.refresh_accuracy_overview()

    def _show_accuracy_context_menu(self, event):
        """
        Right-click an instrument's row to delete just its judging
        history -- the finest granularity available, since
        recognition_agreement.json only stores aggregated per-instrument
        counts. Clicking a family header or sub-row does nothing.
        """
        index = self.accuracy_list.nearest(event.y)
        instrument = self.accuracy_list_row_instrument.get(index)
        if instrument is None:
            return

        self.accuracy_list.selection_clear(0, tk.END)
        self.accuracy_list.selection_set(index)
        self.accuracy_list.activate(index)

        menu = tk.Menu(self.accuracy_list, tearoff=0)
        menu.add_command(
            label=f"Delete {taxonomy_display_name(instrument)}",
            command=lambda: self._delete_recognition_entry(instrument),
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _delete_recognition_entry(self, instrument):
        if not messagebox.askyesno(
            "Delete Judging History",
            f"Permanently delete recorded judging history for "
            f"{taxonomy_display_name(instrument)}? This cannot be undone.",
            parent=self.accuracy_window,
        ):
            return
        stats = self._load_recognition_agreement()
        if instrument not in stats:
            self.refresh_accuracy_overview()
            return
        del stats[instrument]
        try:
            RECOGNITION_AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RECOGNITION_AGREEMENT_PATH, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            self.log(f"Deleted recognition accuracy history for: {instrument}")
        except OSError as exc:
            messagebox.showerror("Could not delete entry", str(exc))
            return
        self.refresh_accuracy_overview()

    def close_accuracy_window(self):
        if self.accuracy_window is not None:
            try:
                self.accuracy_window.destroy()
            except Exception:
                pass
        self.accuracy_window = None
        self.accuracy_list = None
        self.accuracy_overall_label = None
        self.accuracy_list_row_instrument = {}

    def refresh_accuracy_overview(self):
        if self.accuracy_list is None or self.accuracy_overall_label is None:
            return
        self.accuracy_list.delete(0, tk.END)
        self.accuracy_list_row_instrument = {}

        stats = self._load_recognition_agreement()
        if not stats:
            self.accuracy_overall_label.config(
                text="No judged predictions yet -- save or correct a "
                "classification first.",
                fg="#929292",
                font=(self.ui_font, 10),
            )
            return

        # .get(..., 0) rather than direct indexing -- recognition_agreement.json
        # is user-editable; a hand-edited entry missing a key shouldn't
        # crash this window.
        total_agreed = sum(entry.get("agreed", 0) for entry in stats.values())
        total_judged = sum(
            entry.get("agreed", 0) + entry.get("corrected", 0)
            for entry in stats.values()
        )
        overall = total_agreed / total_judged if total_judged else None

        # Same data, aggregated by family_agreed/family_corrected -- a
        # genuinely different number, not a duplicate of the above.
        total_family_agreed = sum(
            entry.get("family_agreed", 0) for entry in stats.values()
        )
        total_family_judged = sum(
            entry.get("family_agreed", 0) + entry.get("family_corrected", 0)
            for entry in stats.values()
        )
        overall_family = (
            total_family_agreed / total_family_judged if total_family_judged else None
        )

        lines = []
        if overall_family is not None:
            lines.append(f"Overall Accuracy (by family): {overall_family * 100:.1f}%")
        if overall is not None:
            lines.append(f"Overall Accuracy (by instrument): {overall * 100:.1f}%")
        self.accuracy_overall_label.config(
            text="\n".join(lines),
            fg="#f8d36a",
            font=(self.ui_font, 13, "bold"),
        )

        # Grouped by family, family accuracy on the header row -- a
        # genuinely different measure than instrument accuracy (violin
        # corrected to viola is a family-level agreement).
        instruments_by_family = {}
        for name in stats:
            try:
                family = family_of(name)
            except ValueError:
                family = None
            instruments_by_family.setdefault(family, []).append(name)

        for family in FAMILY_NAMES:
            names = sorted(instruments_by_family.get(family, []))
            names = [n for n in names if (stats[n].get("agreed", 0) + stats[n].get("corrected", 0)) > 0]
            if not names:
                continue
            # Family-level agreement, not a rollup of instrument-level
            # agreement -- see _record_recognition_agreement's docstring.
            # Falls back to 0/0 (excluded below) for old entries recorded
            # before family_agreed/family_corrected existed.
            family_agreed = sum(stats[n].get("family_agreed", 0) for n in names)
            family_judged = sum(
                stats[n].get("family_agreed", 0) + stats[n].get("family_corrected", 0)
                for n in names
            )
            header = family.upper()
            if family_judged:
                header += (
                    f"  (Family: {family_agreed / family_judged * 100:.1f}%, "
                    f"{family_judged} judged)"
                )
            row_index = self.accuracy_list.size()
            self.accuracy_list.insert(tk.END, header)
            self.accuracy_list.itemconfig(row_index, fg="#f8d36a")
            for name in names:
                entry = stats[name]
                judged = entry.get("agreed", 0) + entry.get("corrected", 0)
                accuracy = entry.get("agreed", 0) / judged
                row_index = self.accuracy_list.size()
                self.accuracy_list.insert(
                    tk.END,
                    f"      {taxonomy_display_name(name)}  "
                    f"({accuracy * 100:.1f}%, {judged} judged)",
                )
                self.accuracy_list_row_instrument[row_index] = name
                self._insert_corrected_to_row(entry)

        # An instrument whose family lookup fails (e.g. the taxonomy was
        # hand-edited after this stat was recorded) still gets shown, just
        # outside any family group, rather than silently dropped.
        for name in sorted(instruments_by_family.get(None, [])):
            entry = stats[name]
            judged = entry.get("agreed", 0) + entry.get("corrected", 0)
            if judged == 0:
                continue
            accuracy = entry.get("agreed", 0) / judged
            row_index = self.accuracy_list.size()
            self.accuracy_list.insert(
                tk.END,
                f"{taxonomy_display_name(name)}  "
                f"({accuracy * 100:.1f}%, {judged} judged)",
            )
            self.accuracy_list_row_instrument[row_index] = name
            self._insert_corrected_to_row(entry)

    def _insert_corrected_to_row(self, entry):
        """
        Shows WHICH instrument(s) a corrected prediction actually got
        corrected to, right under its row -- without this, "Cello (0.0%)"
        sitting under "STRINGS (Family: 100.0%)" looks contradictory. Seeing
        "corrected to: Viola" makes the family/instrument split self-
        explanatory instead of something the user has to infer.
        """
        corrected_to = entry.get("corrected_to")
        if not corrected_to:
            return
        breakdown = ", ".join(
            f"{taxonomy_display_name(name)} ({count})"
            for name, count in sorted(corrected_to.items(), key=lambda kv: -kv[1])
        )
        row_index = self.accuracy_list.size()
        self.accuracy_list.insert(tk.END, f"            → corrected to: {breakdown}")
        self.accuracy_list.itemconfig(row_index, fg="#8a8d90")

    def _load_training_metrics(self):
        """
        Reads models/training_metrics.json, written by train_classifier.py
        at the end of its run (held-out test-set accuracy, overwritten
        every retrain -- see that file's own comment). Returns None if no
        training has produced it yet or it's unreadable.
        """
        if not TRAINING_METRICS_PATH.exists():
            return None
        try:
            with open(TRAINING_METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def open_test_accuracy_overview(self):
        if (
            self.test_accuracy_window is not None
            and self.test_accuracy_window.winfo_exists()
        ):
            self.test_accuracy_window.deiconify()
            self.test_accuracy_window.lift()
            self.refresh_test_accuracy_overview()
            return

        window = tk.Toplevel(self.root)
        self.test_accuracy_window = window
        window.title("Test Set Accuracy")
        window.geometry("380x520")
        window.minsize(340, 400)
        window.configure(bg="#151615")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self.close_test_accuracy_window)
        self._center_toplevel(window)

        title = tk.Label(
            window,
            text="Test Set Accuracy",
            fg="#f4f4f4",
            bg="#171817",
            font=(self.ui_font, 18, "bold"),
        )
        title.pack(fill="x", pady=(0, 8), ipady=10)

        subtitle = tk.Label(
            window,
            text=(
                "From the most recent training run's held-out test set "
                "(never seen during training) -- a model diagnostic, not "
                "how it's performed in real use."
            ),
            fg="#aeb2b5",
            bg="#151615",
            font=(self.ui_font, 9),
            wraplength=340,
            justify="left",
        )
        subtitle.pack(fill="x", padx=16, pady=(0, 8))

        self.test_accuracy_overall_label = tk.Label(
            window,
            text="",
            fg="#f8d36a",
            bg="#151615",
            font=(self.ui_font, 13, "bold"),
        )
        self.test_accuracy_overall_label.pack(fill="x", padx=16, pady=(0, 8))

        body = tk.Frame(window, bg="#151615")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        scroll = tk.Scrollbar(body)
        scroll.pack(side="right", fill="y")
        self.test_accuracy_list = tk.Listbox(
            body,
            bg="#0d0e0d",
            fg="#eeeeee",
            relief="flat",
            bd=0,
            exportselection=False,
            font=(self.ui_font, 11),
            yscrollcommand=scroll.set,
            width=1,
        )
        self.test_accuracy_list.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.test_accuracy_list.yview)
        self.test_accuracy_list.bind(
            "<MouseWheel>",
            lambda event: self.test_accuracy_list.yview_scroll(
                -1 if event.delta > 0 else 1, "units"
            ),
        )

        close_button = self._styled_button(
            window, "Close", self.close_test_accuracy_window
        )
        close_button.pack(fill="x", padx=16, pady=(0, 16), ipady=4)

        self.refresh_test_accuracy_overview()

    def close_test_accuracy_window(self):
        if self.test_accuracy_window is not None:
            try:
                self.test_accuracy_window.destroy()
            except Exception:
                pass
        self.test_accuracy_window = None
        self.test_accuracy_list = None
        self.test_accuracy_overall_label = None

    def refresh_test_accuracy_overview(self):
        if self.test_accuracy_list is None or self.test_accuracy_overall_label is None:
            return
        self.test_accuracy_list.delete(0, tk.END)

        metrics = self._load_training_metrics()
        if metrics is None:
            self.test_accuracy_overall_label.config(
                text="No training results yet -- run training first.",
                fg="#929292",
                font=(self.ui_font, 10),
            )
            return

        overall = metrics.get("overall_accuracy")
        overall_family = metrics.get("overall_family_accuracy")
        lines = []
        if overall_family is not None:
            lines.append(f"Overall Accuracy (by family): {overall_family * 100:.1f}%")
        if overall is not None:
            lines.append(f"Overall Accuracy (by instrument): {overall * 100:.1f}%")
        self.test_accuracy_overall_label.config(
            text="\n".join(lines),
            fg="#f8d36a",
            font=(self.ui_font, 13, "bold"),
        )

        per_instrument = metrics.get("per_instrument_accuracy", {})
        per_family = metrics.get("per_family_accuracy", {})

        # Grouped by family, family accuracy on the header row. Older
        # training_metrics.json files without per_family_accuracy just
        # show headers without a percentage instead of breaking.
        instruments_by_family = {}
        for name in per_instrument:
            try:
                family = family_of(name)
            except ValueError:
                family = None
            instruments_by_family.setdefault(family, []).append(name)

        for family in FAMILY_NAMES:
            instruments = sorted(instruments_by_family.get(family, []))
            if not instruments and family not in per_family:
                continue
            family_accuracy = per_family.get(family)
            header = family.upper()
            if family_accuracy is not None:
                header += f"  (Family: {family_accuracy * 100:.1f}%)"
            row_index = self.test_accuracy_list.size()
            self.test_accuracy_list.insert(tk.END, header)
            self.test_accuracy_list.itemconfig(row_index, fg="#f8d36a")
            for name in instruments:
                accuracy = per_instrument[name]
                self.test_accuracy_list.insert(
                    tk.END,
                    f"      {taxonomy_display_name(name)}  ({accuracy * 100:.1f}%)",
                )

        # An instrument whose family lookup fails (e.g. the taxonomy was
        # hand-edited after this training run) still gets shown, just
        # outside any family group, rather than silently dropped.
        for name in sorted(instruments_by_family.get(None, [])):
            accuracy = per_instrument[name]
            self.test_accuracy_list.insert(
                tk.END, f"{taxonomy_display_name(name)}  ({accuracy * 100:.1f}%)"
            )

    def _build_instrument_taxonomy_choosers(self, parent, class_names, current):
        """
        Linked family -> instrument dropdowns covering both trained
        classes and newly-created-but-not-yet-trained instruments,
        preselected to `current`, with a "+ New Instrument" option.
        Returns a zero-arg callable that reads back the selected raw
        instrument class name, or None if nothing is selected.
        """
        pending_counts = {
            p["name"]: p["count"] for p in self._pending_new_instruments()
        }

        def display_name(name):
            base = taxonomy_display_name(name)
            if name in pending_counts:
                base += f"  (new, {pending_counts[name]}/{NEW_CLASS_MIN_SAMPLES})"
            return base

        state = {"family_to_instruments": {}, "ordered_families": []}

        def rebuild_families():
            known_names = list(class_names) + [
                n for n in pending_counts if n not in class_names
            ]
            family_to_instruments = {}
            for name in known_names:
                try:
                    family = family_of(name)
                except ValueError:
                    # class_names.json still lists this class, but
                    # instrument_taxonomy.json no longer knows it -- skip
                    # rather than crash this dialog.
                    continue
                family_to_instruments.setdefault(family, []).append(name)
            state["family_to_instruments"] = family_to_instruments
            state["ordered_families"] = [
                f for f in FAMILY_NAMES if f in family_to_instruments
            ]

        rebuild_families()

        tk.Label(
            parent,
            text="Family:",
            fg="#aeb2b5",
            bg="#202120",
            font=(self.ui_font, 9),
        ).pack(fill="x", padx=12)
        family_combo = ttk.Combobox(
            parent,
            state="readonly",
            font=(self.ui_font, 11),
        )
        family_combo.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            parent,
            text="Instrument:",
            fg="#aeb2b5",
            bg="#202120",
            font=(self.ui_font, 9),
        ).pack(fill="x", padx=12)
        instrument_combo = ttk.Combobox(
            parent,
            state="readonly",
            font=(self.ui_font, 11),
        )
        instrument_combo.pack(fill="x", padx=12, pady=(0, 6))

        def refresh_instruments(select=None):
            family = state["ordered_families"][family_combo.current()]
            instruments = state["family_to_instruments"][family]
            instrument_combo["values"] = [display_name(i) for i in instruments]
            instrument_combo.current(
                instruments.index(select) if select in instruments else 0
            )

        def refresh_families(select_family=None, select_instrument=None):
            rebuild_families()
            ordered_families = state["ordered_families"]
            family_combo["values"] = [
                taxonomy_display_name(f) for f in ordered_families
            ]
            if not ordered_families:
                # Nothing resolves to a known family -- nothing to select,
                # rather than crash indexing into an empty list below.
                instrument_combo["values"] = []
                return
            target_family = select_family or (
                ordered_families[family_combo.current()]
                if 0 <= family_combo.current() < len(ordered_families)
                else ordered_families[0]
            )
            family_combo.current(
                ordered_families.index(target_family)
                if target_family in ordered_families
                else 0
            )
            refresh_instruments(select=select_instrument)

        family_combo.bind("<<ComboboxSelected>>", lambda _e: refresh_instruments())

        current_family = None
        if current in class_names or current in pending_counts:
            try:
                current_family = family_of(current)
            except ValueError:
                # `current` has no known family -- fall through to "pick
                # any available family" below instead of crashing.
                current_family = None
        if current_family is None and state["ordered_families"]:
            current_family = state["ordered_families"][0]
        refresh_families(select_family=current_family, select_instrument=current)

        def get_selected_class():
            ordered_families = state["ordered_families"]
            family_index = family_combo.current()
            if not ordered_families or not (0 <= family_index < len(ordered_families)):
                return None
            family = ordered_families[family_index]
            instruments = state["family_to_instruments"][family]
            index = instrument_combo.current()
            return instruments[index] if index >= 0 else None

        new_instrument_dialog_holder = {"window": None}

        def open_new_instrument_dialog():
            existing = new_instrument_dialog_holder["window"]
            if existing is not None and existing.winfo_exists():
                existing.lift()
                existing.focus_force()
                return

            sub = tk.Toplevel(parent)
            new_instrument_dialog_holder["window"] = sub
            sub.title("New Instrument")
            sub.configure(bg="#202120")
            sub.transient(parent.winfo_toplevel())
            sub.grab_set()

            tk.Label(
                sub,
                text="New Instrument",
                fg="#f8d36a",
                bg="#202120",
                font=(self.ui_font, 12, "bold"),
            ).pack(fill="x", padx=16, pady=(14, 4))
            tk.Label(
                sub,
                text=(
                    "Just registers the category -- it won't actually be "
                    "recognized until enough corrections have filed audio "
                    f"under it (about {NEW_CLASS_MIN_SAMPLES} samples) and "
                    "the classifier is retrained."
                ),
                fg="#aeb2b5",
                bg="#202120",
                font=(self.ui_font, 9),
                justify="left",
                wraplength=300,
            ).pack(fill="x", padx=16, pady=(0, 10))

            tk.Label(
                sub, text="Name:", fg="#aeb2b5", bg="#202120", font=(self.ui_font, 9)
            ).pack(fill="x", padx=16)
            name_entry = tk.Entry(sub, font=(self.ui_font, 11))
            name_entry.pack(fill="x", padx=16, pady=(0, 10))
            name_entry.focus_set()

            tk.Label(
                sub, text="Family:", fg="#aeb2b5", bg="#202120", font=(self.ui_font, 9)
            ).pack(fill="x", padx=16)
            new_family_marker = "+ New Family..."
            family_sub_combo = ttk.Combobox(
                sub,
                values=[taxonomy_display_name(f) for f in FAMILY_NAMES]
                + [new_family_marker],
                state="readonly",
                font=(self.ui_font, 11),
            )
            family_sub_combo.current(0)
            family_sub_combo.pack(fill="x", padx=16, pady=(0, 6))

            new_family_entry = tk.Entry(sub, font=(self.ui_font, 11))

            def on_family_change(_e=None):
                if family_sub_combo.get() == new_family_marker:
                    new_family_entry.pack(fill="x", padx=16, pady=(0, 10))
                else:
                    new_family_entry.pack_forget()

            family_sub_combo.bind("<<ComboboxSelected>>", on_family_change)

            def confirm():
                raw_name = name_entry.get().strip().lower().replace(" ", "_")
                if not raw_name:
                    messagebox.showerror(
                        "Missing name", "Enter an instrument name.", parent=sub
                    )
                    return
                is_new_family = family_sub_combo.get() == new_family_marker
                if is_new_family:
                    family_name = (
                        new_family_entry.get().strip().lower().replace(" ", "_")
                    )
                    if not family_name:
                        messagebox.showerror(
                            "Missing family name",
                            "Enter a name for the new family.",
                            parent=sub,
                        )
                        return
                else:
                    family_name = FAMILY_NAMES[family_sub_combo.current()]

                try:
                    add_instrument(raw_name, family_name, new_family=is_new_family)
                except ValueError as exc:
                    messagebox.showerror(
                        "Could not create instrument", str(exc), parent=sub
                    )
                    return

                pending_counts[raw_name] = 0
                sub.destroy()
                refresh_families(select_family=family_name, select_instrument=raw_name)
                self.log(
                    f"Created new instrument '{raw_name}' under family "
                    f"'{family_name}'. It needs about {NEW_CLASS_MIN_SAMPLES} "
                    "corrected samples before retraining will recognize it."
                )

            tk.Button(
                sub,
                text="Create",
                command=confirm,
                bg="#f8d36a",
                fg="#111111",
                activebackground="#ffdf8c",
                relief="flat",
                font=(self.ui_font, 10, "bold"),
                cursor="hand2",
            ).pack(fill="x", padx=16, pady=(4, 16), ipady=5)

            self._center_toplevel(sub, relative_to=parent.winfo_toplevel())

        # macOS ignores a tk.Button's custom bg (always renders its native
        # light face), so fg must stay dark here to actually be readable --
        # same technique as safe_dialogs.py's secondary buttons.
        tk.Button(
            parent,
            text="+ New Instrument",
            command=open_new_instrument_dialog,
            fg="#111111",
            activeforeground="#000000",
            relief="raised",
            bd=1,
            highlightbackground="#2b2c2d",
            highlightcolor="#2b2c2d",
            highlightthickness=2,
            font=(self.ui_font, 9),
            cursor="hand2",
        ).pack(fill="x", padx=12, pady=(0, 12), ipady=3)

        return get_selected_class

    def close_sound_manager(self):
        if self.sound_manager_window is not None:
            try:
                self.sound_manager_window.destroy()
            except Exception:
                pass
        self.sound_manager_window = None
        self.sound_manager_taxonomy_list = None
        self.sound_manager_sound_list = None
        self.sound_manager_details_label = None
        self.save_sound_button = None

    def refresh_sound_manager(
        self,
        select_family_folder=None,
        select_instrument_folder=None,
        select_sound_folder=None,
    ):
        if (
            self.sound_manager_window is None
            or not self.sound_manager_window.winfo_exists()
            or self.sound_manager_taxonomy_list is None
        ):
            return

        previous = self._selected_manager_taxonomy_row()
        wanted_family = select_family_folder or (
            previous.get("family_folder") if previous else None
        )
        wanted_instrument = select_instrument_folder or (
            previous.get("instrument_folder") if previous else None
        )

        self.sound_manager_taxonomy_rows = []
        self.sound_manager_taxonomy_list.delete(0, tk.END)

        selected_index = 0
        for family in list_taxonomy():
            family_row_index = self.sound_manager_taxonomy_list.size()
            family_count = sum(i["sound_count"] for i in family["instruments"])
            self.sound_manager_taxonomy_rows.append(
                {
                    "kind": "family",
                    "family_folder": family["family_folder"],
                    "instrument_folder": None,
                    "name": family["family_folder"],
                }
            )
            self.sound_manager_taxonomy_list.insert(
                tk.END, f"{family['family_folder'].upper()}  ({family_count})"
            )
            self.sound_manager_taxonomy_list.itemconfig(family_row_index, fg="#f8d36a")
            if family["family_folder"] == wanted_family and wanted_instrument is None:
                selected_index = family_row_index

            for instrument in family["instruments"]:
                row_index = self.sound_manager_taxonomy_list.size()
                self.sound_manager_taxonomy_rows.append(
                    {
                        "kind": "instrument",
                        "family_folder": instrument["family_folder"],
                        "instrument_folder": instrument["instrument_folder"],
                        "name": instrument["instrument_folder"],
                    }
                )
                self.sound_manager_taxonomy_list.insert(
                    tk.END,
                    f"      {taxonomy_display_name(instrument['instrument_folder'])}"
                    f"  ({instrument['sound_count']})",
                )
                if (
                    instrument["family_folder"] == wanted_family
                    and instrument["instrument_folder"] == wanted_instrument
                ):
                    selected_index = row_index

        if self.sound_manager_taxonomy_rows:
            self.sound_manager_taxonomy_list.selection_set(selected_index)
            self.sound_manager_taxonomy_list.activate(selected_index)
            self.sound_manager_taxonomy_list.see(selected_index)

        self.refresh_manager_sounds(select_sound_folder=select_sound_folder)

    def on_manager_taxonomy_row_selected(self):
        self.refresh_manager_sounds()

    def refresh_manager_sounds(self, select_sound_folder=None):
        if self.sound_manager_sound_list is None:
            return

        row = self._selected_manager_taxonomy_row()
        self.sound_manager_sound_records = []
        self.sound_manager_sound_list.delete(0, tk.END)

        if row is None:
            self.update_manager_sound_details()
            return

        self.sound_manager_sound_records = list_sounds(
            row["family_folder"], row["instrument_folder"]
        )
        selected_index = None

        for index, sound in enumerate(self.sound_manager_sound_records):
            detected = sound.get("detected_instrument_class") or "unknown"
            self.sound_manager_sound_list.insert(
                tk.END,
                f"{sound['name']}    [{taxonomy_display_name(detected)}]",
            )
            if sound.get("folder_name") == select_sound_folder:
                selected_index = index

        if self.sound_manager_sound_records:
            if selected_index is None:
                selected_index = 0
            self.sound_manager_sound_list.selection_set(selected_index)
            self.sound_manager_sound_list.activate(selected_index)
            self.sound_manager_sound_list.see(selected_index)

        self.update_manager_sound_details()

    def update_manager_sound_details(self):
        if self.sound_manager_details_label is None:
            return
        sound = self._selected_manager_sound()
        if sound is None:
            self.sound_manager_details_label.config(text="No sounds in this category.")
            return

        detected = sound.get("detected_instrument_class") or "unknown"
        # created_at is stored as a full ISO timestamp (YYYY-MM-DDTHH:MM:SS)
        # for precision, but the hour/minute/second and the ISO date order
        # are more than anyone needs to see here -- just MM/DD/YYYY.
        created_iso = sound.get("created_at", "unknown").split("T")[0]
        created_parts = created_iso.split("-")
        if len(created_parts) == 3:
            year, month, day = created_parts
            created = f"{month}/{day}/{year}"
        else:
            created = created_iso
        self.sound_manager_details_label.config(
            fg="#aeb2b5",
            text=(
                f"Name: {sound.get('name', 'Unnamed')}    "
                f"Instrument: {taxonomy_display_name(detected)}    Created: {created}"
            ),
        )

    def ask_string_dialog(self, title, prompt, initial="", parent=None):
        """
        Custom replacement for tkinter.simpledialog.askstring. On some
        macOS versions, popping simpledialog's native modal window right
        after creating another new window can race with a system
        GameControllerUI/Tk autorelease-pool bug and crash the whole
        process (EXC_BREAKPOINT inside Tcl_DoOneEvent) -- this avoids the
        stock dialog machinery entirely as a workaround.
        """
        parent = parent or self.root
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.configure(bg="#202120")
        dialog.transient(parent)
        dialog.resizable(False, False)

        tk.Label(
            dialog,
            text=prompt,
            fg="#eeeeee",
            bg="#202120",
            font=(self.ui_font, 11),
            justify="left",
            wraplength=320,
        ).pack(padx=16, pady=(16, 8), fill="x")

        entry_var = tk.StringVar(value=initial)
        entry = tk.Entry(
            dialog, textvariable=entry_var, font=(self.ui_font, 11), width=32
        )
        entry.pack(padx=16, pady=(0, 12), fill="x")
        entry.focus_set()
        entry.select_range(0, tk.END)

        result = {"value": None}

        def confirm(event=None):
            result["value"] = entry_var.get()
            dialog.destroy()

        def cancel(event=None):
            dialog.destroy()

        button_row = tk.Frame(dialog, bg="#202120")
        button_row.pack(fill="x", padx=16, pady=(0, 16))
        tk.Button(
            button_row,
            text="Cancel",
            command=cancel,
            bg="#d7d7d7",
            fg="#111111",
            font=(self.ui_font, 10),
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(
            button_row,
            text="OK",
            command=confirm,
            bg="#d7d7d7",
            fg="#111111",
            font=(self.ui_font, 10, "bold"),
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

        entry.bind("<Return>", confirm)
        dialog.bind("<Escape>", cancel)

        dialog.update_idletasks()
        self._center_toplevel(dialog, relative_to=parent)
        dialog.grab_set()
        dialog.wait_window()

        return result["value"]

    def manager_save_current_sound(self):
        if not LEARNED_PARAMS.exists():
            messagebox.showerror(
                "No learned timbre",
                "Run System first, then save the learned timbre.",
                parent=self.sound_manager_window,
            )
            return
        if self.run_in_progress:
            messagebox.showerror(
                "System busy",
                "Wait for the current run to finish before saving -- it's "
                "still writing the learned timbre files.",
                parent=self.sound_manager_window,
            )
            return

        info = self.parse_summary()
        detected_class = info.get("predicted_class", "")
        if detected_class in UNCLASSIFIED_VALUES:
            messagebox.showerror(
                "Missing classification",
                "The current CNN classification is unavailable. Run System first.",
                parent=self.sound_manager_window,
            )
            return

        name = self.ask_string_dialog(
            "Save Current Timbre",
            f"CNN classification: {taxonomy_display_name(detected_class)}\n\nEnter a timbre name:",
            parent=self.sound_manager_window,
        )
        if name is None:
            return

        try:
            saved = save_sound(
                sound_name=name,
                detected_instrument_class=detected_class,
                learned_params_path=LEARNED_PARAMS,
                preview_wav_path=(
                    RECONSTRUCTED_AUDIO if RECONSTRUCTED_AUDIO.exists() else None
                ),
                source_audio_path=(INPUT_AUDIO if INPUT_AUDIO.exists() else None),
                last_correction_file=(
                    Path(self._current_input_correction_file)
                    if self._current_input_correction_file is not None
                    else None
                ),
            )
            self.refresh_sound_manager(
                select_family_folder=saved.parent.parent.name,
                select_instrument_folder=saved.parent.name,
                select_sound_folder=saved.name,
            )
            # Saved without correcting classification first -- counts as a
            # real-world CNN-got-it-right outcome. If it had been
            # corrected, fix_current_classification already recorded the
            # disagreement, so don't double-count it here.
            if self._current_input_correction_file is None:
                self._record_recognition_agreement(detected_class, agreed=True)
            self.log(f"Saved timbre '{name.strip()}' as {detected_class}.")
            messagebox.showinfo(
                "Timbre saved",
                f"Saved '{name.strip()}' under "
                f"{taxonomy_display_name(detected_class)}.",
                parent=self.sound_manager_window,
            )
        except Exception as exc:
            messagebox.showerror("Could not save timbre", str(exc))

    def _show_sound_manager_context_menu(self, event):
        index = self.sound_manager_sound_list.nearest(event.y)
        if index < 0 or index >= len(self.sound_manager_sound_records):
            return
        self.sound_manager_sound_list.selection_clear(0, tk.END)
        self.sound_manager_sound_list.selection_set(index)
        self.sound_manager_sound_list.activate(index)
        self.update_manager_sound_details()

        menu = tk.Menu(self.sound_manager_sound_list, tearoff=0)
        menu.add_command(label="Rename", command=self.manager_rename_sound)
        menu.add_command(label="Delete", command=self.manager_delete_sound)
        menu.tk_popup(event.x_root, event.y_root)

    def manager_rename_sound(self):
        sound = self._selected_manager_sound()
        if sound is None:
            messagebox.showerror("No timbre selected", "Select a timbre first.")
            return
        name = self.ask_string_dialog(
            "Rename Timbre",
            "Enter the new timbre name:",
            initial=sound["name"],
            parent=self.sound_manager_window,
        )
        if name is None:
            return
        try:
            folder = rename_sound(
                sound["family_folder"],
                sound["instrument_folder"],
                sound["folder_name"],
                name,
            )
            self.refresh_sound_manager(
                select_family_folder=sound["family_folder"],
                select_instrument_folder=sound["instrument_folder"],
                select_sound_folder=folder.name,
            )
            self.log(f"Renamed timbre to: {name.strip()}")
        except Exception as exc:
            messagebox.showerror("Could not rename timbre", str(exc))

    def reset_whole_system(self):
        """
        Full reset for starting over entirely: deletes every saved timbre,
        every correction_*.wav filed under data/ by a classification
        correction, and all Recognition Accuracy judging history.
        Deliberately does not touch the original baseline dataset or the
        trained classifier model itself.
        """
        if not messagebox.askyesno(
            "Reset Whole System",
            "This will permanently delete:\n"
            "  • Every saved timbre\n"
            "  • Every correction audio file added to the classifier's "
            "training data\n"
            "  • All Recognition Accuracy judging history\n\n"
            "The original training dataset and the trained classifier "
            "model are not affected.\n\nThis cannot be undone. Continue?",
            parent=self.root,
        ):
            return

        if not messagebox.askyesno(
            "Are You Sure?",
            "Really delete everything listed above? There is no undo.",
            parent=self.root,
        ):
            return

        timbre_count = 0
        if SOUND_LIBRARY_ROOT.exists():
            for sound_dir in SOUND_LIBRARY_ROOT.glob("*/*/*"):
                if sound_dir.is_dir():
                    timbre_count += 1
            shutil.rmtree(SOUND_LIBRARY_ROOT)
            SOUND_LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)

        correction_count = 0
        if DATA_DIR.exists():
            for correction_file in DATA_DIR.glob("*/*/correction_*.wav"):
                correction_file.unlink()
                correction_count += 1

        if RECOGNITION_AGREEMENT_PATH.exists():
            RECOGNITION_AGREEMENT_PATH.unlink()

        # Whatever the main panel currently has loaded/tracked no longer
        # points at anything real.
        self.loaded_timbre_location = None
        self._current_input_correction_file = None
        self.reset_run_state()

        self.log(
            f"Reset whole system: deleted {timbre_count} saved timbre(s), "
            f"{correction_count} correction audio file(s), and all "
            "Recognition Accuracy history."
        )
        self.refresh_sound_manager()
        if self.accuracy_window is not None and self.accuracy_window.winfo_exists():
            self.refresh_accuracy_overview()

        messagebox.showinfo(
            "Reset Complete",
            f"Deleted {timbre_count} saved timbre(s) and {correction_count} "
            "correction audio file(s). Recognition Accuracy history cleared.",
            parent=self.root,
        )

    def manager_delete_sound(self):
        sound = self._selected_manager_sound()
        if sound is None:
            messagebox.showerror("No timbre selected", "Select a timbre first.")
            return
        if not messagebox.askyesno(
            "Delete Timbre",
            f"Permanently delete '{sound['name']}'?",
            parent=self.sound_manager_window,
        ):
            return
        try:
            delete_sound(
                sound["family_folder"], sound["instrument_folder"], sound["folder_name"]
            )
            # Deleting the sound currently on the keyboard must not leave
            # loaded_timbre_location pointing at a folder that no longer
            # exists.
            if self.loaded_timbre_location == (
                sound["family_folder"],
                sound["instrument_folder"],
                sound["folder_name"],
            ):
                self.loaded_timbre_location = None
            self.refresh_sound_manager(
                select_family_folder=sound["family_folder"],
                select_instrument_folder=sound["instrument_folder"],
            )
            self.log(f"Deleted timbre: {sound['name']}")
        except Exception as exc:
            messagebox.showerror("Could not delete timbre", str(exc))

    def manager_fix_classification(self):
        if (
            self._manager_fix_classification_chooser is not None
            and self._manager_fix_classification_chooser.winfo_exists()
        ):
            self._manager_fix_classification_chooser.lift()
            self._manager_fix_classification_chooser.focus_force()
            return

        sound = self._selected_manager_sound()
        if sound is None:
            messagebox.showerror("No timbre selected", "Select a timbre first.")
            return

        if not sound.get("source_audio_path"):
            messagebox.showerror(
                "No source audio",
                "This sound has no source_audio.wav saved, so it can't be "
                "added to the classifier's training data.",
                parent=self.sound_manager_window,
            )
            return

        class_names = self._load_class_names()

        if not class_names:
            messagebox.showerror(
                "No classes available",
                "Could not read models/class_names.json. Train the "
                "classifier at least once first.",
                parent=self.sound_manager_window,
            )
            return

        chooser = tk.Toplevel(self.sound_manager_window)
        self._manager_fix_classification_chooser = chooser
        chooser.title("Correct CNN Instrument Label")
        chooser.geometry("380x520")
        chooser.configure(bg="#202120")
        chooser.transient(self.sound_manager_window)
        self._center_toplevel(chooser, relative_to=self.sound_manager_window)
        chooser.grab_set()

        tk.Label(
            chooser,
            text="Correct CNN Instrument Label",
            fg="#f8d36a",
            bg="#202120",
            font=(self.ui_font, 13, "bold"),
        ).pack(fill="x", padx=12, pady=(14, 2))

        tk.Label(
            chooser,
            text=(
                "Corrects which instrument the CNN classifier thinks this "
                "timbre is -- which controls its MIDI pitch range -- and "
                "adds the recording to that instrument's training data. "
                "The timbre is also refiled into the corrected instrument's "
                "library folder to match."
            ),
            fg="#aeb2b5",
            bg="#202120",
            font=(self.ui_font, 9),
            justify="left",
            wraplength=350,
        ).pack(fill="x", padx=12, pady=(0, 10))

        current = sound.get("detected_instrument_class") or "unknown"
        tk.Label(
            chooser,
            text=f"'{sound['name']}' is currently labeled: {current}",
            fg="#eeeeee",
            bg="#202120",
            font=(self.ui_font, 11, "bold"),
            justify="left",
        ).pack(fill="x", padx=12, pady=(0, 8))

        get_selected_class = self._build_instrument_taxonomy_choosers(
            chooser, class_names, current
        )

        is_loaded_timbre = self.loaded_timbre_location == (
            sound["family_folder"],
            sound["instrument_folder"],
            sound["folder_name"],
        )

        def confirm_fix():
            correct_class = get_selected_class()
            if not correct_class or correct_class == current:
                # Nothing changed -- avoid correct_sound_class creating a
                # redundant duplicate training file and counting toward
                # the retrain-prompt threshold for no reason.
                chooser.destroy()
                return

            # correct_sound_class is slow enough that an impatient second
            # click could fire before the folder move it does completes,
            # failing with a confusing "Sound not found" error.
            self._set_button_enabled(save_button, False)
            try:
                destination, training_copy = correct_sound_class(
                    sound["family_folder"],
                    sound["instrument_folder"],
                    sound["folder_name"],
                    correct_class,
                    source_audio_path=Path(sound["source_audio_path"]),
                    data_dir=DATA_DIR,
                )

                # Different instruments render to different MIDI pitch
                # ranges/synthesis presets, so if this timbre is the one
                # currently on the keyboard, re-render it for the
                # corrected class instead of leaving the old range up.
                if is_loaded_timbre and LEARNED_PARAMS.exists():
                    self.loaded_timbre_location = (
                        destination.parent.parent.name,
                        destination.parent.name,
                        destination.name,
                    )
                    midi_notes = self._render_notes_for_class(
                        correct_class, "Re-rendering notes..."
                    )
                    self.refresh_outputs()
                    self.enable_output_buttons()
                    self.count_label.config(text=f"Notes: {len(midi_notes)}")

                self.refresh_sound_manager(
                    select_family_folder=destination.parent.parent.name,
                    select_instrument_folder=destination.parent.name,
                    select_sound_folder=destination.name,
                )
                self.log(
                    f"Fixed classification of '{sound['name']}' to "
                    f"'{correct_class}'. Copied to {training_copy}. "
                    "Click 'Retrain Classifier' when ready to learn from it."
                )

                if self._maybe_prompt_retrain(
                    correct_class, chooser
                ) or self._maybe_prompt_new_instrument_retrain(correct_class, chooser):
                    do_retrain()
                    return

                messagebox.showinfo(
                    "Classification fixed",
                    f"'{sound['name']}' is now labeled '{correct_class}' and "
                    "its recording was added to the training data.\n\n"
                    "Click 'Retrain Classifier' below to have the model "
                    "learn from it now, or close this window to do it "
                    "later.",
                    parent=chooser,
                )
                # Keep the dialog open (instead of closing it) so the user
                # can retrain right here instead of hunting for the button
                # back in the Timbre Manager.
                save_button.pack_forget()
                retrain_button.pack(fill="x", padx=12, pady=(0, 12), ipady=5)
            except Exception as exc:
                messagebox.showerror(
                    "Could not fix classification", str(exc), parent=chooser
                )
                self._set_button_enabled(save_button, True)

        def do_retrain():
            chooser.destroy()
            self.retrain_classifier_threaded()

        save_button = self._cnn_button(chooser, "Save Correction", confirm_fix)
        save_button.pack(fill="x", padx=12, pady=(0, 12), ipady=5)

        retrain_button = self._cnn_button(chooser, "Retrain Classifier", do_retrain)
        # Not packed yet -- only shown once a correction has been saved.

    def retrain_classifier_threaded(self):
        if getattr(self, "classifier_retrain_in_progress", False):
            messagebox.showerror(
                "Already retraining",
                "A classifier retrain is already running in the background.",
                parent=self.sound_manager_window,
            )
            return

        thin_pending = [
            p
            for p in self._pending_new_instruments()
            if p["count"] < NEW_CLASS_MIN_SAMPLES
        ]
        if thin_pending:
            details = "\n".join(
                f"- {taxonomy_display_name(p['name'])}: "
                f"{p['count']}/{NEW_CLASS_MIN_SAMPLES} samples"
                for p in thin_pending
            )
            if not messagebox.askyesno(
                "New instruments still thin on data",
                "These newly created instruments don't have much training "
                "data yet and likely won't be recognized well after this "
                f"retrain:\n\n{details}\n\n"
                "Retrain anyway?",
                parent=self.sound_manager_window,
            ):
                return

        self.classifier_retrain_in_progress = True
        self.log("Retraining classifier in the background...")

        thread = threading.Thread(target=self.retrain_classifier_worker, daemon=True)
        thread.start()
        if self.retrain_queue_after_id is None:
            self.poll_retrain_queue()

    def retrain_classifier_worker(self):
        try:
            self.retrain_process = subprocess.Popen(
                [sys.executable, str(TRAIN_CLASSIFIER_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            output_lines = []
            if self.retrain_process.stdout is not None:
                for line in self.retrain_process.stdout:
                    output_lines.append(line)
            return_code = self.retrain_process.wait()
            success = return_code == 0
            tail = "\n".join(output_lines[-15:]) if output_lines else ""
        except Exception as exc:
            success = False
            tail = str(exc)
        finally:
            self.retrain_process = None

        self.retrain_queue.put((success, tail))

    def poll_retrain_queue(self):
        try:
            while True:
                success, tail = self.retrain_queue.get_nowait()
                self.finish_retrain_classifier(success, tail)
        except queue.Empty:
            pass

        self.retrain_queue_after_id = self.root.after(100, self.poll_retrain_queue)

    def finish_retrain_classifier(self, success, tail):
        self.classifier_retrain_in_progress = False
        if success:
            self.log("Classifier retrain finished.")
            messagebox.showinfo(
                "Retrain complete", "Classifier retrain finished successfully."
            )
        else:
            self.log(f"Classifier retrain failed:\n{tail}")
            messagebox.showerror(
                "Retrain failed",
                "Classifier retrain failed. Check the log for details.",
            )

    def manager_preview_sound(self):
        sound = self._selected_manager_sound()
        if sound is None:
            messagebox.showerror("No timbre selected", "Select a timbre first.")
            return
        preview = sound.get("preview_path")
        if not preview:
            messagebox.showerror("No preview", "This timbre has no preview.wav file.")
            return
        self.play_one_shot(Path(preview))

    def _render_notes_for_class(
        self, detected_class, loading_message="Rendering notes..."
    ):
        """
        Run render_notes.py against the currently learned timbre
        (LEARNED_PARAMS) using detected_class's MIDI range/synthesis
        preset, replacing whatever's in RENDERED_DIR. Raises on failure.
        Shared by loading a saved timbre and by classification-correction
        flows that need the keyboard to reflect the corrected
        instrument's real range, since different instruments render to
        different pitch ranges/presets (mode_selector.get_mode_config).
        """
        # Center the rendered range on this timbre's own learned pitch, not
        # the instrument class's aggregate range (see
        # mode_selector._range_start_from_f0).
        source_f0_hz = None
        if LEARNED_PARAMS.exists():
            try:
                with open(LEARNED_PARAMS, "r", encoding="utf-8") as f:
                    source_f0_hz = json.load(f).get("f0_hz")
            except (OSError, json.JSONDecodeError):
                source_f0_hz = None

        mode_config = get_mode_config(detected_class, source_f0_hz=source_f0_hz)
        midi_notes = mode_config["midi_notes"]
        render_args = mode_config["render_args"]
        postprocess = mode_config.get("postprocess", {})

        # Render into a scratch directory and only replace RENDERED_DIR on
        # success, so a failed render doesn't leave the keyboard with zero
        # playable notes.
        temp_dir = RENDERED_DIR.parent / f"{RENDERED_DIR.name}_rendering_tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.show_loading(loading_message)
        self.root.update_idletasks()

        render_command = [
            sys.executable,
            str(RENDER_SCRIPT),
            "--params",
            str(LEARNED_PARAMS),
            "--out_dir",
            str(temp_dir),
            "--transient_gain",
            str(render_args["transient_gain"]),
        ]
        # Only presets that opt in (mode_selector.py's vocoder_hybrid) get
        # the WORLD-vocoder path -- only sounds right on a continuous
        # sustained tone, not a plucked/struck attack.
        if mode_config.get("vocoder_hybrid", False) and INPUT_AUDIO.exists():
            render_command += ["--source_audio", str(INPUT_AUDIO)]

        # Optional -- only acoustic_piano sets these (see
        # ACOUSTIC_PIANO_RENDER_ARGS in mode_selector.py).
        if "inharmonicity" in render_args:
            render_command += ["--inharmonicity", str(render_args["inharmonicity"])]
        if "unison_voices" in render_args:
            render_command += ["--unison_voices", str(render_args["unison_voices"])]
        if "unison_detune_cents" in render_args:
            render_command += [
                "--unison_detune_cents",
                str(render_args["unison_detune_cents"]),
            ]

        render_command += ["--midi", *[str(note) for note in midi_notes]]

        try:
            result = subprocess.run(
                render_command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            self.hide_loading()

        if result.stdout:
            for line in result.stdout.splitlines():
                self.log(line)
        if result.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("render_notes.py failed.")

        if RENDERED_DIR.exists():
            shutil.rmtree(RENDERED_DIR)
        temp_dir.rename(RENDERED_DIR)

        # Matches run_system.py's own render step (e.g. mode_selector's
        # BASS_CLICK) -- without this, loading a saved timbre or
        # re-rendering after Fix Classification would silently miss it.
        if postprocess.get("bass_click", False):
            self.show_loading(loading_message)
            self.root.update_idletasks()
            try:
                enhance_result = subprocess.run(
                    [
                        sys.executable,
                        str(ENHANCE_SCRIPT),
                        "--folder",
                        str(RENDERED_DIR),
                        "--amount",
                        str(postprocess.get("click_amount", 0.45)),
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            finally:
                self.hide_loading()

            if enhance_result.stdout:
                for line in enhance_result.stdout.splitlines():
                    self.log(line)
            if enhance_result.returncode != 0:
                raise RuntimeError("enhance_rendered_notes.py failed.")

        return midi_notes

    def _silence_preview_before_load(self):
        """
        Called before Load does anything else, so an active or in-flight
        "Play Preview" clip can't collide with it and pop: fades out
        anything already playing (PreviewPlayer.stop()), bumps
        preview_generation so a load that's still decoding skips playback
        instead of racing past the fade, then fully closes the stream so
        it doesn't compete for CPU with Load's own work.
        """
        self.preview_player.stop()
        self.preview_generation += 1
        time.sleep(PreviewPlayer.FADE_SECONDS + 0.02)
        self.preview_player.close()

    def manager_load_sound(self):
        self._silence_preview_before_load()

        sound = self._selected_manager_sound()
        if sound is None:
            messagebox.showerror("No timbre selected", "Select a timbre first.")
            return
        if self.run_in_progress:
            messagebox.showerror("System busy", "Wait for the current run to finish.")
            return

        try:
            self._load_sound_record(sound)
            # Do not interrupt the workflow with a modal popup after loading.
            # The loaded timbre is shown in the main status area and log instead.
            self.close_sound_manager()
        except Exception as exc:
            self.hide_loading()
            self.log(f"Load timbre error: {exc}")
            messagebox.showerror("Could not load timbre", str(exc))

    def _load_sound_record(self, sound):
        """
        Loads `sound` (a dict with at least family_folder/instrument_
        folder/folder_name, e.g. from _selected_manager_sound() or
        list_sounds()) onto the keyboard. Raises on failure -- callers
        decide how to report that. Shared by the Timbre Manager's Load
        button and the main panel's quick-switch dropdown (jumping
        straight to another saved sound under the same instrument).
        """
        record = load_sound(
            sound["family_folder"],
            sound["instrument_folder"],
            sound["folder_name"],
        )
        detected_class = record.get("detected_instrument_class")
        if detected_class in UNCLASSIFIED_VALUES:
            raise ValueError(
                "This timbre does not contain a supported CNN instrument class."
            )

        if self.engine is not None:
            self.engine.stop_all()

        TIMBRE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(record["params_path"]), LEARNED_PARAMS)

        preview_path = record.get("preview_path")
        if preview_path and Path(preview_path).exists():
            shutil.copy2(Path(preview_path), RECONSTRUCTED_AUDIO)

        source_audio_path = record.get("source_audio_path")
        if source_audio_path and Path(source_audio_path).exists():
            INPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source_audio_path), INPUT_AUDIO)
            self.audio_uploaded = True

        midi_notes = self._render_notes_for_class(detected_class, "Loading Timbre...")

        self.loaded_timbre_location = (
            sound["family_folder"],
            sound["instrument_folder"],
            sound["folder_name"],
        )
        self.refresh_outputs()
        self.enable_output_buttons()
        self.prediction_label.config(
            text=f"Loaded timbre: {record['name']} | Instrument: {taxonomy_display_name(detected_class)}"
        )
        # Not tied to parse_summary()/INPUT_AUDIO in this state -- fix
        # this timbre's classification from the Timbre Manager instead.
        self._set_prediction_clickable(False)
        self.count_label.config(text=f"Notes: {len(midi_notes)}")
        self.log(f"Loaded saved timbre: {record['name']} ({detected_class})")

    def _bordered_panel(self, parent, title, bg="#202120"):
        """
        Self-drawn replacement for tk.LabelFrame(text=..., labelanchor="n")
        -- native LabelFrame border/title rendering is unreliable on macOS
        Aqua (inconsistent, distorted border segments). Draws the border
        via highlightthickness (reliable cross-platform, same technique
        already used for the Predicted Class chip) and the centered title
        the same way as _section_divider, so panel titles and in-panel
        section dividers render identically. Returns the panel frame;
        pack/grid children into it directly.
        """
        panel = tk.Frame(
            parent,
            bg=bg,
            highlightthickness=1,
            highlightbackground="#3a3b3c",
            highlightcolor="#3a3b3c",
        )
        self._section_divider(panel, title, bg=bg, top_pad=5)
        return panel

    def _section_divider(self, parent, label, bg="#202120", top_pad=6):
        """Caption embedded in a horizontal rule, marking where one group
        of controls within a panel ends and the next begins (e.g. Input vs
        System) -- same groupbox look as a LabelFrame's own centered
        title/border, just with tighter padding since a panel can hold
        several of these."""
        row = tk.Frame(parent, bg=bg)
        row.pack(fill="x", padx=12, pady=(top_pad, 3))
        tk.Frame(row, bg="#0c0c0c", height=1).pack(
            side="left", fill="x", expand=True, pady=7
        )
        tk.Label(
            row,
            text=f" {label} ",
            fg="#eeeeee",
            bg=bg,
            font=(self.ui_font, 11, "bold"),
        ).pack(side="left")
        tk.Frame(row, bg="#0c0c0c", height=1).pack(
            side="left", fill="x", expand=True, pady=7
        )

    def _styled_button(self, parent, text, command):
        """Shared look for the app's plain action buttons (Run System,
        Timbre Manager, Save/Load Timbre, Play Input/Output, Tutorial,
        etc.) so they all read as one consistent button style."""
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=(self.ui_font, 9),
            bg="#d7d7d7",
            fg="#111111",
            # Without this, Tk falls back to its own washed-out default
            # disabled-text color, which reads as nearly invisible on
            # macOS's native (always-light) button face.
            disabledforeground="#6b6b6b",
            activebackground="#eeeeee",
            relief="raised",
            bd=1,
            cursor="hand2",
        )

    def _set_button_enabled(self, button, enabled, **extra_config):
        """
        Toggle a button's clickability and cursor together -- Tk doesn't
        change a disabled button's cursor on its own, so a plain
        state="disabled" still shows the hand cursor on hover.
        """
        if button is None:
            return
        button.config(
            state="normal" if enabled else "disabled",
            cursor="hand2" if enabled else "arrow",
            **extra_config,
        )

    def _center_toplevel(self, window, relative_to=None):
        """
        Centers `window` over `relative_to` (defaults to self.root). Call
        this last, after the window's size is settled -- after
        geometry("WxH") for fixed-size windows, or after all widgets are
        packed/gridded for windows that size to their content.
        """
        parent = relative_to or self.root
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        if width <= 1 or height <= 1:
            width = window.winfo_reqwidth()
            height = window.winfo_reqheight()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def discard_generated_notes(self):
        """
        The main panel's "Refresh" button. Deletes the just-generated
        rendered notes and resets the panel to "no timbre loaded".
        Deliberately does not call _record_recognition_agreement --
        discarding isn't a real-world judgment of the CNN's prediction,
        just "never mind, start over."
        """
        if RENDERED_DIR.exists():
            shutil.rmtree(RENDERED_DIR, ignore_errors=True)
            self.log("Refresh: deleted the generated notes for this instrument.")

        self.reset_run_state()

    def reset_run_state(self):
        """
        Clear classification / rendered-notes display so a freshly dropped
        audio file doesn't keep showing the PREVIOUS file's Run System
        results until Run System actually executes for this one.
        """
        if self.engine:
            self.engine.stop_all()

        self.prediction_label.config(text="Predicted Class: ----")
        self._set_prediction_clickable(False)
        self.count_label.config(text="Notes: 0")

        self.wav_by_midi = {}
        self.min_midi = None
        self.max_midi = None

        self.draw_keyboard()
        self.update_octave_label()

        # Run System/Refresh need audio uploaded this session, not just a
        # leftover file on disk from a previous one.
        self._set_button_enabled(self.run_button, self.audio_uploaded)
        self._set_button_enabled(self.refresh_button, self.audio_uploaded)
        self._set_button_enabled(self.play_input_button, False)
        self._set_button_enabled(self.play_output_button, False)
        self.timbre_ready = False
        self.loaded_timbre_location = None
        self._current_input_correction_file = None
        self._set_button_enabled(self.save_sound_button, False)
        self._set_button_enabled(self.save_timbre_button, False)

    def enable_output_buttons(self):
        if INPUT_AUDIO.exists():
            self._set_button_enabled(self.play_input_button, True)
        if RECONSTRUCTED_AUDIO.exists():
            self._set_button_enabled(self.play_output_button, True)
        if LEARNED_PARAMS.exists():
            self.timbre_ready = True
            self._set_button_enabled(self.save_timbre_button, True)
            self._set_button_enabled(self.save_sound_button, True)

    def parse_summary(self):
        result = {
            "predicted_class": "not loaded",
            "predicted_family": "",
            "notes": "0",
        }
        if not SUMMARY_FILE.exists():
            return result

        text = SUMMARY_FILE.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("predicted_class:"):
                result["predicted_class"] = line.split(":", 1)[1].strip()
            elif line.startswith("predicted_family:"):
                result["predicted_family"] = line.split(":", 1)[1].strip()
            elif line.startswith("render_midi_notes_count:"):
                result["notes"] = line.split(":", 1)[1].strip()

        return result

    def refresh_outputs(self):
        if not self.audio_uploaded:
            return

        if self.engine:
            self.engine.stop_all()

        info = self.parse_summary()
        if info["predicted_class"] in UNCLASSIFIED_VALUES:
            label_text = f"Predicted Class: {info['predicted_class']}"
        else:
            label_text = f"Predicted Class: {taxonomy_display_name(info['predicted_class'])}"
            if info["predicted_family"]:
                label_text += f" ({taxonomy_display_name(info['predicted_family'])})"
        self.prediction_label.config(text=label_text)
        self._set_prediction_clickable(
            info["predicted_class"] not in UNCLASSIFIED_VALUES
        )
        self.count_label.config(text=f"Notes: {info['notes']}")

        self.wav_by_midi = {}

        if RENDERED_DIR.exists():
            for wav in sorted(RENDERED_DIR.glob("*.wav")):
                midi = parse_midi_from_filename(wav)
                if midi is not None:
                    self.wav_by_midi[midi] = wav

        if self.wav_by_midi:
            self.min_midi = min(self.wav_by_midi.keys())
            self.max_midi = max(self.wav_by_midi.keys())

            self.base_midi = self._default_base_midi_for_range()
            self.home_base_midi = self.base_midi
            self.octave_shift = 0

            try:
                if self.engine is None:
                    self.engine = AudioEngine()
                self.engine.load_notes(self.wav_by_midi)
                self.log(
                    "Audio engine ready. Clean mixer + transparent limiter enabled."
                )
            except Exception as e:
                self.log(f"Audio engine error: {e}")
                messagebox.showerror("Audio engine error", str(e))

        self.draw_keyboard()
        self.update_octave_label()
        self.log("Outputs refreshed.")

    def _set_prediction_clickable(self, enabled):
        """
        Only let the "Predicted Class" label be clicked to fix when it's
        showing a fresh classification for the currently loaded audio --
        not a loaded-from-library timbre, whose classification is fixed
        through the Timbre Manager instead.
        """
        self.classification_available = enabled
        if enabled:
            self.prediction_label.config(
                cursor="hand2",
                fg="#f8d36a",
                highlightbackground="#f8d36a",
                highlightcolor="#f8d36a",
                bg="#242017",
            )
            self.prediction_fix_hint.config(text="(click to correct)", cursor="hand2")
        else:
            self.prediction_label.config(
                cursor="arrow",
                fg="#8d8d8d",
                highlightbackground="#3a3b3c",
                highlightcolor="#3a3b3c",
                bg="#242017",
            )
            self.prediction_fix_hint.config(text="", cursor="arrow")

    def _on_prediction_hover_enter(self, event=None):
        if self.classification_available:
            self.prediction_label.config(bg="#3a331f")

    def _on_prediction_hover_leave(self, event=None):
        if self.classification_available:
            self.prediction_label.config(bg="#242017")

    def fix_current_classification(self, event=None):
        """
        Let the user click the "Predicted Class" label in the status bar
        and correct it right there, without first saving the timbre to the
        library. Adds the current input audio to the classifier's training
        data under the corrected class, same as the Timbre Manager's
        "Correct CNN Label" action.
        """
        if not self.classification_available:
            return

        if (
            self._fix_classification_chooser is not None
            and self._fix_classification_chooser.winfo_exists()
        ):
            self._fix_classification_chooser.lift()
            self._fix_classification_chooser.focus_force()
            return

        info = self.parse_summary()
        current = info.get("predicted_class", "")
        if current in UNCLASSIFIED_VALUES:
            messagebox.showerror(
                "No classification yet",
                "Run System first, then you can correct the classification.",
            )
            return

        if not INPUT_AUDIO.exists():
            messagebox.showerror(
                "Missing input audio",
                "The original input audio is missing, so a correction "
                "can't be saved.",
            )
            return

        class_names = self._load_class_names()

        if not class_names:
            messagebox.showerror(
                "No classes available",
                "Could not read models/class_names.json. Train the "
                "classifier at least once first.",
            )
            return

        chooser = tk.Toplevel(self.root)
        self._fix_classification_chooser = chooser
        chooser.title("Correct CNN Instrument Label")
        chooser.geometry("380x520")
        chooser.configure(bg="#202120")
        chooser.transient(self.root)
        self._center_toplevel(chooser)
        chooser.grab_set()

        tk.Label(
            chooser,
            text="Correct CNN Instrument Label",
            fg="#f8d36a",
            bg="#202120",
            font=(self.ui_font, 13, "bold"),
        ).pack(fill="x", padx=12, pady=(14, 2))

        tk.Label(
            chooser,
            text=(
                "Corrects which instrument the CNN classifier thinks the "
                "currently loaded audio is -- which controls its MIDI "
                "pitch range -- and adds the recording to that "
                "instrument's training data."
            ),
            fg="#aeb2b5",
            bg="#202120",
            font=(self.ui_font, 9),
            justify="left",
            wraplength=350,
        ).pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(
            chooser,
            text=f"Currently classified as: {current}",
            fg="#eeeeee",
            bg="#202120",
            font=(self.ui_font, 11, "bold"),
            justify="left",
        ).pack(fill="x", padx=12, pady=(0, 8))

        get_selected_class = self._build_instrument_taxonomy_choosers(
            chooser, class_names, current
        )

        def confirm_fix():
            correct_class = get_selected_class()
            if not correct_class or correct_class == current:
                chooser.destroy()
                return

            # See the equivalent guard in manager_fix_classification's
            # confirm_fix -- avoids a redundant duplicate correction from
            # an impatient second click.
            self._set_button_enabled(save_button, False)
            try:
                if self._current_input_correction_file is not None:
                    previous_path = Path(self._current_input_correction_file)
                    if previous_path.exists():
                        previous_path.unlink()

                destination = add_correction_to_training_data(
                    INPUT_AUDIO, correct_class, DATA_DIR
                )
                self._current_input_correction_file = str(destination)
                self._set_summary_predicted_class(correct_class)
                # Record as a real-world disagreement keyed by what the
                # CNN predicted. Family guess can still be right even when
                # the instrument isn't (violin -> viola is still strings).
                try:
                    family_agreed = family_of(current) == family_of(correct_class)
                except ValueError:
                    family_agreed = False
                self._record_recognition_agreement(
                    current,
                    agreed=False,
                    family_agreed=family_agreed,
                    corrected_to=correct_class,
                )

                # Different instruments use different MIDI ranges/presets
                # (mode_selector.get_mode_config), so re-render, not just
                # relabel.
                if LEARNED_PARAMS.exists():
                    midi_notes = self._render_notes_for_class(
                        correct_class, "Re-rendering notes..."
                    )
                    self.refresh_outputs()
                    self.enable_output_buttons()
                    self.count_label.config(text=f"Notes: {len(midi_notes)}")
                else:
                    info = self.parse_summary()
                    if info["predicted_class"] in UNCLASSIFIED_VALUES:
                        label_text = f"Predicted Class: {info['predicted_class']}"
                    else:
                        label_text = (
                            f"Predicted Class: "
                            f"{taxonomy_display_name(info['predicted_class'])}"
                        )
                        if info["predicted_family"]:
                            label_text += (
                                f" ({taxonomy_display_name(info['predicted_family'])})"
                            )
                    self.prediction_label.config(text=label_text)

                self.log(
                    f"Corrected classification to '{correct_class}'. "
                    f"Copied to {destination}. Click 'Retrain Classifier' "
                    "when ready to learn from it."
                )

                if self._maybe_prompt_retrain(
                    correct_class, chooser
                ) or self._maybe_prompt_new_instrument_retrain(correct_class, chooser):
                    do_retrain()
                    return

                messagebox.showinfo(
                    "Classification corrected",
                    f"Classification set to '{correct_class}' and the "
                    "recording was added to the training data.\n\n"
                    "Click 'Retrain Classifier' below to have the model "
                    "learn from it now, or close this window to do it "
                    "later.",
                    parent=chooser,
                )
                # Keep the dialog open (instead of closing it) so the user
                # can retrain right here instead of hunting for the button
                # back in the Timbre Manager.
                save_button.pack_forget()
                retrain_button.pack(fill="x", padx=12, pady=(0, 12), ipady=5)
            except Exception as exc:
                messagebox.showerror(
                    "Could not correct classification", str(exc), parent=chooser
                )
                self._set_button_enabled(save_button, True)

        def do_retrain():
            chooser.destroy()
            self.retrain_classifier_threaded()

        save_button = self._cnn_button(chooser, "Save Correction", confirm_fix)
        save_button.pack(fill="x", padx=12, pady=(0, 12), ipady=5)

        retrain_button = self._cnn_button(chooser, "Retrain Classifier", do_retrain)
        # Not packed yet -- only shown once a correction has been saved.

    def _set_summary_predicted_class(self, correct_class):
        if not SUMMARY_FILE.exists():
            return
        text = SUMMARY_FILE.read_text(encoding="utf-8")
        text = PREDICTED_CLASS_LINE_RE.sub(
            f"predicted_class: {correct_class}", text, count=1
        )
        text = PREDICTED_FAMILY_LINE_RE.sub(
            f"predicted_family: {family_of(correct_class)}", text, count=1
        )
        SUMMARY_FILE.write_text(text, encoding="utf-8")

    def update_octave_label(self):
        if self.min_midi is None:
            self.octave_label.config(text="Base: -- | Octave: --")
            return

        sign = "+" if self.octave_shift > 0 else ""
        self.octave_label.config(
            text=f"Base: {midi_to_note_name(self.base_midi)} | Octave: {sign}{self.octave_shift}"
        )

    def _update_octave_label_for_midi_note(self, note):
        """
        Keeps the "Base: X | Octave: Y" label reflecting a MIDI keyboard's
        own octave-shift buttons too. Those buttons send no distinct MIDI
        message, so there's no event to react to -- instead, treat every
        played note as revealing the current octave, without touching
        self.base_midi/self.octave_shift (which drive the computer
        keyboard's own mapping and shouldn't be affected by this).
        """
        if self.min_midi is None:
            return
        current_c = note - (note % 12)
        octave_shift = round((current_c - self.home_base_midi) / 12)
        sign = "+" if octave_shift > 0 else ""
        self.octave_label.config(
            text=f"Base: {midi_to_note_name(current_c)} | Octave: {sign}{octave_shift}"
        )

    def shift_octave(self, direction: int):
        if not self.wav_by_midi:
            return

        # Steps between whole octaves anchored to C, but a rendered range
        # rarely starts/ends exactly on a C -- add the true min/max as one
        # extra step past the last C on each end so every rendered note
        # stays reachable regardless of where the range actually starts.
        candidates = sorted(
            midi for midi in self.wav_by_midi if midi % 12 == 0
        )
        if not candidates or self.min_midi < candidates[0]:
            candidates.insert(0, self.min_midi)
        # KEY_TO_OFFSET spans 18 semitones (1.5 octaves) upward from the
        # base, so the top step only needs to move the base if even that
        # isn't enough to cover the highest rendered note.
        top_step = max(self.min_midi, self.max_midi - 18)
        if not candidates or top_step > candidates[-1]:
            candidates.append(top_step)
        candidates = sorted(set(candidates))

        if direction > 0:
            options = [midi for midi in candidates if midi > self.base_midi]
            new_base = min(options) if options else self.base_midi
        else:
            options = [midi for midi in candidates if midi < self.base_midi]
            new_base = max(options) if options else self.base_midi

        if new_base == self.base_midi:
            self.log(f"Octave limit reached at {midi_to_note_name(self.base_midi)}.")
            return

        self.base_midi = new_base
        self.octave_shift = round((self.base_midi - self.home_base_midi) / 12)
        self.update_octave_label()

        direction_text = "UP" if direction > 0 else "DOWN"
        self.log(
            f"Octave {direction_text}: {self.octave_shift:+d} | "
            f"base MIDI {self.base_midi} ({midi_to_note_name(self.base_midi)})"
        )

    def note_on(self, midi: int, velocity: int = 127, source: str = "gui"):
        if self.engine is None:
            if source != "midi":
                self.log("Audio engine not ready.")
            return

        ok = self.engine.note_on(midi, velocity=velocity)
        if not ok:
            if source != "midi":
                self.log(f"No rendered note for MIDI {midi}")
            return

        self.press_key_visual(midi)

        if source != "midi":
            self.log(f"Note on: MIDI {midi} {midi_to_note_name(midi)}")

    def note_off(self, midi: int, source: str = "gui"):
        if self.engine:
            self.engine.note_off(midi)

        self.release_key_visual(midi)

        if source != "midi":
            self.log(f"Note off: MIDI {midi} {midi_to_note_name(midi)}")

    def on_keyboard_press(self, event):
        key = normalize_key(event.keysym)

        # If this key had a delayed release, this is OS auto-repeat.
        # Cancel release and do NOT retrigger.
        if key in self.release_jobs:
            try:
                self.root.after_cancel(self.release_jobs[key])
            except Exception:
                pass
            self.release_jobs.pop(key, None)
            return

        if key == "z":
            if key not in self.pressed_keyboard_keys:
                self.pressed_keyboard_keys[key] = None
                self.shift_octave(-1)
            return

        if key == "x":
            if key not in self.pressed_keyboard_keys:
                self.pressed_keyboard_keys[key] = None
                self.shift_octave(1)
            return

        if key not in KEY_TO_OFFSET:
            return

        if key in self.pressed_keyboard_keys:
            return

        midi = self.base_midi + KEY_TO_OFFSET[key]
        self.pressed_keyboard_keys[key] = midi
        self.note_on(midi)

    def on_keyboard_release(self, event):
        key = normalize_key(event.keysym)

        if key not in self.pressed_keyboard_keys:
            return

        if key not in KEY_TO_OFFSET:
            self.pressed_keyboard_keys.pop(key, None)
            return

        midi = self.pressed_keyboard_keys[key]

        def delayed_release(k=key):
            self.release_jobs.pop(k, None)
            current_midi = self.pressed_keyboard_keys.pop(k, None)
            if current_midi is not None:
                self.note_off(current_midi)

        self.release_jobs[key] = self.root.after(120, delayed_release)

    def press_key_visual(self, midi: int):
        info = self.key_items.get(midi)
        if not info:
            return

        self.keyboard_canvas.itemconfig(
            info["rect"], fill="#f8d36a" if info["type"] == "white" else "#444444"
        )
        self.keyboard_canvas.coords(
            info["rect"], info["x0"], info["y0"] + 6, info["x1"], info["y1"] + 6
        )
        self.keyboard_canvas.coords(
            info["text"], (info["x0"] + info["x1"]) / 2, info["text_y"] + 6
        )

    def release_key_visual(self, midi: int):
        info = self.key_items.get(midi)
        if not info:
            return

        self.keyboard_canvas.itemconfig(info["rect"], fill=info["normal_fill"])
        self.keyboard_canvas.coords(
            info["rect"], info["x0"], info["y0"], info["x1"], info["y1"]
        )
        self.keyboard_canvas.coords(
            info["text"], (info["x0"] + info["x1"]) / 2, info["text_y"]
        )

    def find_midi_under_mouse(self, event):
        items = self.keyboard_canvas.find_overlapping(
            event.x, event.y, event.x, event.y
        )

        # Topmost item wins; black keys are drawn above white keys.
        for item in reversed(items):
            tags = self.keyboard_canvas.gettags(item)
            for tag in tags:
                if tag.startswith("key_"):
                    try:
                        midi = int(tag.split("_", 1)[1])
                        if midi in self.wav_by_midi:
                            return midi
                    except ValueError:
                        pass

        return None

    def start_mouse_note(self, midi: int):
        if midi == self.mouse_down_note:
            return

        if self.mouse_down_note is not None:
            self.note_off(self.mouse_down_note)

        self.mouse_down_note = midi

        if midi is not None:
            self.note_on(midi)

    def on_mouse_drag(self, event):
        midi = self.find_midi_under_mouse(event)
        if midi is not None:
            self.start_mouse_note(midi)

    def on_mouse_release_anywhere(self, event):
        if self.mouse_down_note is not None:
            self.note_off(self.mouse_down_note)
            self.mouse_down_note = None

    def on_mouse_leave_keyboard(self, event):
        if self.mouse_down_note is not None:
            self.note_off(self.mouse_down_note)
            self.mouse_down_note = None

    def bind_canvas_key_events(self, midi: int):
        tag = f"key_{midi}"

        self.keyboard_canvas.tag_bind(
            tag,
            "<ButtonPress-1>",
            lambda event, m=midi: self.start_mouse_note(m),
        )

        # Release and drag are handled globally by the canvas:
        # <ButtonRelease-1> and <B1-Motion>.

    def draw_keyboard(self):
        self.keyboard_canvas.delete("all")
        self.key_items = {}

        if not self.wav_by_midi:
            self.keyboard_canvas.create_text(
                650,
                130,
                text="No rendered notes. Run system first.",
                fill="#eeeeee",
                font=("Arial", 16, "bold"),
            )
            return

        midi_notes = sorted(self.wav_by_midi.keys())
        min_midi = min(midi_notes)
        max_midi = max(midi_notes)

        white_notes = [
            midi
            for midi in range(min_midi, max_midi + 1)
            if midi % 12 in WHITE_PITCH_CLASSES
        ]

        # Use the actual visible canvas width and center the keyboard.
        # The previous fixed 1220 px width left a large empty area on wide windows.
        self.root.update_idletasks()
        visible_width = max(
            900,
            self.keyboard_canvas.winfo_width() - 16,
        )

        natural_keyboard_width = len(white_notes) * 56
        keyboard_width = min(
            visible_width,
            max(980, natural_keyboard_width),
        )

        x_offset = max(
            8,
            (visible_width - keyboard_width) / 2,
        )

        white_w = keyboard_width / len(white_notes)
        white_h = 225
        black_w = white_w * 0.62
        black_h = 138

        canvas_height = 270
        self.keyboard_canvas.config(
            scrollregion=(
                0,
                0,
                visible_width,
                canvas_height,
            )
        )

        white_x = {}
        current_x = x_offset

        for midi in white_notes:
            x0 = current_x
            x1 = current_x + white_w
            y0 = 22
            y1 = 22 + white_h
            white_x[midi] = x0

            normal_fill = "#ffffff" if midi in self.wav_by_midi else "#aaaaaa"

            rect = self.keyboard_canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=normal_fill,
                outline="#111111",
                width=2,
                tags=(f"key_{midi}",),
            )

            label = midi_to_note_name(midi)
            text_y = y1 - 20
            text = self.keyboard_canvas.create_text(
                (x0 + x1) / 2,
                text_y,
                text=label,
                fill="#111111",
                font=("Arial", 10, "bold"),
                tags=(f"key_{midi}",),
            )

            self.key_items[midi] = {
                "type": "white",
                "rect": rect,
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text_y": text_y,
                "normal_fill": normal_fill,
            }

            if midi in self.wav_by_midi:
                self.bind_canvas_key_events(midi)

            current_x += white_w

        for midi in range(min_midi, max_midi + 1):
            if midi % 12 not in BLACK_PITCH_CLASSES:
                continue

            prev_white = midi - 1
            while prev_white % 12 not in WHITE_PITCH_CLASSES:
                prev_white -= 1

            if prev_white not in white_x:
                continue

            x_center = white_x[prev_white] + white_w
            x0 = x_center - black_w / 2
            x1 = x_center + black_w / 2
            y0 = 22
            y1 = 22 + black_h

            normal_fill = "#050505" if midi in self.wav_by_midi else "#555555"

            rect = self.keyboard_canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=normal_fill,
                outline="#222222",
                width=2,
                tags=(f"key_{midi}",),
            )

            text_y = y1 - 18
            text = self.keyboard_canvas.create_text(
                (x0 + x1) / 2,
                text_y,
                text=midi_to_note_name(midi),
                fill="#eeeeee",
                font=("Arial", 8, "bold"),
                tags=(f"key_{midi}",),
            )

            self.key_items[midi] = {
                "type": "black",
                "rect": rect,
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text_y": text_y,
                "normal_fill": normal_fill,
            }

            if midi in self.wav_by_midi:
                self.bind_canvas_key_events(midi)

    def run_system_threaded(self):
        if self.run_in_progress:
            return

        if not self.audio_uploaded or not INPUT_AUDIO.exists():
            messagebox.showerror(
                "No audio uploaded",
                "Drag and drop an audio file first, then press Run System.",
            )
            return

        if self.engine:
            try:
                self.engine.close()
            except Exception as exc:
                self.log(f"Audio engine close warning: {exc}")
            self.engine = None

        self.run_in_progress = True
        self._set_button_enabled(self.run_button, False, text="Running...")

        self.show_loading("Running System...")

        self.log("=" * 80)
        self.log("Running full system...")
        self.log("=" * 80)

        self.run_thread = threading.Thread(
            target=self.run_system_worker,
            daemon=True,
        )
        self.run_thread.start()

        self.root.after(50, self.poll_run_queue)

    def run_system_worker(self):
        try:
            if not RUN_SYSTEM.is_file():
                raise FileNotFoundError(
                    "Backend script not found.\n"
                    f"Expected: {RUN_SYSTEM}\n"
                    f"Detected project root: {PROJECT_ROOT}\n"
                    "Place this GUI inside the project, normally as gui/app.py."
                )

            self.run_queue.put(("log", f"Python: {sys.executable}"))
            self.run_queue.put(("log", f"Project root: {PROJECT_ROOT}"))
            self.run_queue.put(("log", f"Backend: {RUN_SYSTEM}"))

            self.run_process = subprocess.Popen(
                [sys.executable, "-u", str(RUN_SYSTEM)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            total_steps = None
            train_start_time = None
            train_start_step = None
            rendered_total = None
            rendered_count = 0

            if self.run_process.stdout is not None:
                for raw_line in self.run_process.stdout:
                    line = raw_line.rstrip()
                    self.run_queue.put(("log", line))

                    if total_steps is None:
                        m = TRAIN_ARGS_STEPS_RE.search(line)
                        if m:
                            total_steps = int(m.group(1))
                            self.run_queue.put(
                                (
                                    "progress",
                                    {
                                        "phase": "Training timbre model...",
                                        "percent": 0,
                                        "sub": f"Step 0/{total_steps}",
                                    },
                                )
                            )
                            continue

                    if total_steps is not None:
                        m = STEP_LINE_RE.match(line)
                        if m:
                            current_step = int(m.group(1))
                            now = time.time()
                            if train_start_time is None:
                                train_start_time = now
                                train_start_step = current_step
                            elapsed = now - train_start_time
                            done_steps = current_step - train_start_step
                            percent = min(100, current_step / total_steps * 100)
                            eta_text = ""
                            if done_steps > 0 and elapsed > 0:
                                rate = done_steps / elapsed
                                remaining = max(0, total_steps - current_step)
                                if rate > 0:
                                    eta_text = (
                                        f" | Estimated time: {int(remaining / rate)}s"
                                    )
                            self.run_queue.put(
                                (
                                    "progress",
                                    {
                                        "phase": "Training timbre model...",
                                        "percent": percent,
                                        "sub": (
                                            f"Step {current_step}/{total_steps}"
                                            f"{eta_text}"
                                        ),
                                    },
                                )
                            )
                            continue

                    if total_steps is not None and (
                        "Early stopping at step" in line
                        or line.startswith("Training steps:")
                    ):
                        self.run_queue.put(
                            (
                                "progress",
                                {
                                    "phase": "Training timbre model...",
                                    "percent": 100,
                                    "sub": "Finalizing...",
                                },
                            )
                        )
                        continue

                    m = RENDERING_COUNT_RE.search(line)
                    if m:
                        rendered_total = int(m.group(1))
                        rendered_count = 0
                        self.run_queue.put(
                            (
                                "progress",
                                {
                                    "phase": "Rendering notes...",
                                    "percent": 0,
                                    "sub": f"0/{rendered_total}",
                                },
                            )
                        )
                        continue

                    if rendered_total and SAVED_WAV_RE.match(line):
                        rendered_count += 1
                        percent = min(100, rendered_count / rendered_total * 100)
                        self.run_queue.put(
                            (
                                "progress",
                                {
                                    "phase": "Rendering notes...",
                                    "percent": percent,
                                    "sub": f"{rendered_count}/{rendered_total}",
                                },
                            )
                        )
                        continue

                    if line.startswith("Enhanced bass click"):
                        self.run_queue.put(
                            (
                                "progress",
                                {
                                    "phase": "Post-processing...",
                                    "percent": 100,
                                    "sub": "",
                                },
                            )
                        )
                        continue

            return_code = self.run_process.wait()
            self.run_queue.put(("finished", return_code))

        except Exception as exc:
            self.run_queue.put(("exception", str(exc)))

    def poll_run_queue(self):
        try:
            while True:
                message_type, payload = self.run_queue.get_nowait()

                if message_type == "log":
                    self.log(payload)

                elif message_type == "progress":
                    self.update_loading_progress(
                        payload["phase"],
                        percent=payload.get("percent"),
                        sub_text=payload.get("sub"),
                    )

                elif message_type == "finished":
                    self.finish_system_run(payload)
                    return

                elif message_type == "exception":
                    self.finish_system_run(
                        return_code=None,
                        error_message=payload,
                    )
                    return

        except queue.Empty:
            pass

        if self.run_in_progress:
            self.root.after(50, self.poll_run_queue)

    def finish_system_run(self, return_code, error_message=None):
        self.run_in_progress = False
        self.run_process = None
        self._set_button_enabled(self.run_button, True, text="Run System")

        self.hide_loading()

        if error_message is not None:
            self.log(f"System launch error: {error_message}")
            messagebox.showerror(
                "System launch failed",
                error_message,
            )
            return

        if return_code == 0:
            self.log("Done.")
            self.refresh_outputs()
            self.enable_output_buttons()
            messagebox.showinfo(
                "Done",
                "System completed.",
            )
        else:
            self.log(f"Failed: {return_code}")
            messagebox.showerror(
                "System failed",
                "The backend stopped with an error. Check the log.",
            )

    def on_close(self):
        self.close_sound_manager()
        self.disconnect_midi_input(update_status=False)
        self.close_midi_manager_window()
        self.close_data_overview_window()
        self.close_accuracy_window()
        self.close_test_accuracy_window()

        if self.meter_after_id is not None:
            try:
                self.root.after_cancel(self.meter_after_id)
            except Exception:
                pass
            self.meter_after_id = None

        if self.midi_autoconnect_after_id is not None:
            try:
                self.root.after_cancel(self.midi_autoconnect_after_id)
            except Exception:
                pass
            self.midi_autoconnect_after_id = None
        if self.midi_queue_after_id is not None:
            try:
                self.root.after_cancel(self.midi_queue_after_id)
            except Exception:
                pass
            self.midi_queue_after_id = None
        if self.midi_note_queue_after_id is not None:
            try:
                self.root.after_cancel(self.midi_note_queue_after_id)
            except Exception:
                pass
            self.midi_note_queue_after_id = None

        # Wake the persistent scan worker out of its blocking queue .get()
        # so it exits promptly (see _midi_scan_worker_loop).
        self.midi_scan_worker_generation += 1
        self.midi_scan_request_queue.put((False, -1))

        if self.run_process and self.run_process.poll() is None:
            try:
                self.run_process.terminate()
            except Exception:
                pass

        if self.retrain_process and self.retrain_process.poll() is None:
            try:
                self.retrain_process.terminate()
            except Exception:
                pass

        if self.retrain_queue_after_id is not None:
            try:
                self.root.after_cancel(self.retrain_queue_after_id)
            except Exception:
                pass
            self.retrain_queue_after_id = None

        self.hide_loading()

        if self.engine:
            try:
                self.engine.close()
            except Exception:
                pass

        if self.preview_player:
            try:
                self.preview_player.close()
            except Exception:
                pass

        self.root.destroy()


def main():
    if sd is None:
        print("Missing dependency: sounddevice")
        print("Install it with:")
        print("pip install sounddevice")
        return

    if TkinterDnD is None:
        # Drag-and-drop is optional -- fall back to a plain Tk root instead
        # of refusing to launch; the file browser still works without it.
        print("tkinterdnd2 not installed -- drag & drop will be unavailable.")
        print("Install it with: python -m pip install tkinterdnd2")
        root = tk.Tk()
    else:
        root = TkinterDnD.Tk()

    SynthGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
