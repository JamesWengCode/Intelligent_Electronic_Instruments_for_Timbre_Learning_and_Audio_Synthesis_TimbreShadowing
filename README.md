# Timbre Shadowing

A desktop app that learns the timbre of a recorded instrument and lets you
play it back as a full, pitch-shifted playable keyboard.

1. **Classify** -- a hierarchical CNN (`src/train_classifier.py`,
   `src/predict_instrument.py`) identifies the instrument family and
   specific instrument of an input recording.
2. **Learn the timbre** -- a harmonic + noise DDSP-style synthesizer
   (`src/train_timbre.py`) fits the recording's spectral envelope,
   transient noise, and brightness decay.
3. **Render playable notes** -- `src/render_notes.py` generates a full
   36-note range from the learned timbre, using a WORLD-vocoder pitch-shift
   hybrid for bowed strings/wind/brass (reusing the real recorded timbre
   instead of pure synthesis) and DDSP harmonic synthesis elsewhere.
4. **Play** -- `gui/app.py` is a Tkinter GUI: a playable on-screen/MIDI
   keyboard, EQ, reverb, delay, ADSR envelope controls, and a saved-sound
   library organized by the same instrument taxonomy as the classifier.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python3 gui/app.py
```

Or run the pipeline directly from the command line on `input/target.wav`:

```bash
python3 src/run_system.py
```

## Project Layout

```
gui/app.py                    Playable GUI (keyboard, EQ, FX, sound library)
src/train_classifier.py       Trains the hierarchical instrument classifier
src/predict_instrument.py     Runs the trained classifier on one recording
src/train_timbre.py           Fits the DDSP harmonic timbre model to a recording
src/render_notes.py           Renders the learned timbre across a 36-note range
src/enhance_rendered_notes.py Post-processing (transient layering, etc.)
src/run_system.py             One-command CLI pipeline (classify -> learn -> render)
src/mode_selector.py          Per-instrument-family synthesis/render presets
src/sound_library.py          Saved-sound library, organized by instrument taxonomy
src/instrument_taxonomy.py    Shared family/instrument taxonomy
models/                       Pretrained classifier weights and metadata
data/                         Training audio (not tracked -- see data/README.md)
saved_timbres/                User-saved sounds (generated at runtime)
outputs/                      Pipeline run outputs (generated at runtime)
```

## Data

`data/` (training audio, ~2 GB) is not tracked in this repo. See
[`data/README.md`](data/README.md) for the expected folder layout and where
the training audio came from. The classifier can still run without it --
`models/` already ships the pretrained weights.

## Team

Timbre Shadowing is one half of a two-part capstone project built by a
four-person team. James Weng and Wen-Ting Chen were primarily
responsible for Timbre Shadowing (this repository); Chia-Hui Yang and
I-Pei Chen were primarily responsible for the project's other half,
[TimbreEdge](https://github.com/Kuan801/TimbreEdge), a real-time
embedded instrument on a Teensy 4.1. All four members contributed to
system integration, testing, and the project report.
