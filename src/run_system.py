"""
run_system.py

One-command software pipeline with instrument-specific generation modes.

Run:
python3 src/run_system.py
"""

import json
import time
import shutil
import subprocess
import sys
from pathlib import Path

from mode_selector import get_mode_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_AUDIO = PROJECT_ROOT / "input" / "target.wav"

PREDICT_SCRIPT = PROJECT_ROOT / "src" / "predict_instrument.py"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train_timbre.py"
RENDER_SCRIPT = PROJECT_ROOT / "src" / "render_notes.py"
ENHANCE_SCRIPT = PROJECT_ROOT / "src" / "enhance_rendered_notes.py"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LATEST_RUN = OUTPUT_ROOT / "latest_run"
TIMBRE_DIR = LATEST_RUN / "timbre_learning"
RENDERED_DIR = LATEST_RUN / "rendered_notes"
LEARNED_PARAMS = TIMBRE_DIR / "learned_params.json"


def check_file(path: Path, message: str):
    if not path.exists():
        raise FileNotFoundError(f"{message}\nMissing: {path}")


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    print("\nRunning:")
    print(" ".join(str(x) for x in command))
    print("-" * 80)

    # -u (unbuffered): a piped child's stdout is block-buffered otherwise,
    # so progress lines arrive in one burst at exit instead of live.
    if command and command[0] == sys.executable:
        command = [command[0], "-u", *command[1:]]

    start_time = time.perf_counter()

    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines = []
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)

    returncode = process.wait()
    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    stdout_text = "".join(output_lines)

    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode,
            command,
            output=stdout_text,
        )

    return subprocess.CompletedProcess(
        command, returncode, stdout=stdout_text, stderr=""
    )


def prepare_output_folder():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if LATEST_RUN.exists():
        shutil.rmtree(LATEST_RUN)

    TIMBRE_DIR.mkdir(parents=True, exist_ok=True)
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)


def predict_instrument() -> tuple[str, str, str]:
    result = run_command(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--audio",
            str(INPUT_AUDIO),
        ]
    )

    predicted_instrument = None
    predicted_family = None

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("instrument:"):
            predicted_instrument = line.split(":", 1)[1].strip()
        elif line.startswith("family:"):
            predicted_family = line.split(":", 1)[1].strip()

    if predicted_instrument is None or predicted_family is None:
        raise RuntimeError(
            "predict_instrument.py did not print both an instrument and a "
            "family prediction."
        )

    classifier_output = result.stdout

    return predicted_instrument, predicted_family, classifier_output


def train_timbre(mode_config: dict):
    train_args = mode_config["train_args"]

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--target",
        str(INPUT_AUDIO),
        "--out_dir",
        str(TIMBRE_DIR),
        "--f0",
        "auto",
        "--steps",
        str(train_args["steps"]),
        "--residual_strength",
        str(train_args["residual_strength"]),
        "--transient_energy_weight",
        str(train_args["transient_energy_weight"]),
        "--transient_late_weight",
        str(train_args["transient_late_weight"]),
        "--transient_smooth_weight",
        str(train_args["transient_smooth_weight"]),
    ]

    # Optional -- only presets that need it set these (BOWED/WIND_TRAIN_ARGS
    # in mode_selector.py); others keep train_timbre.py's own defaults.
    if "amp_smooth_kernel" in train_args:
        command += ["--amp_smooth_kernel", str(train_args["amp_smooth_kernel"])]
    if "harmonic_smooth_kernel" in train_args:
        command += [
            "--harmonic_smooth_kernel",
            str(train_args["harmonic_smooth_kernel"]),
        ]

    run_command(command)

    check_file(
        LEARNED_PARAMS,
        "Training completed but learned_params.json was not created.",
    )


def render_notes(mode_config: dict):
    midi_notes = mode_config["midi_notes"]
    render_args = mode_config["render_args"]

    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--params",
        str(LEARNED_PARAMS),
        "--out_dir",
        str(RENDERED_DIR),
        "--transient_gain",
        str(render_args["transient_gain"]),
    ]

    # Only presets that opt in (mode_selector.py's vocoder_hybrid) get the
    # WORLD-vocoder pitch-shift path -- it only sounds right on a
    # continuous sustained tone, not a plucked/struck attack.
    if mode_config.get("vocoder_hybrid", False):
        command += ["--source_audio", str(INPUT_AUDIO)]

    # Optional -- only acoustic_piano sets these (see
    # ACOUSTIC_PIANO_RENDER_ARGS in mode_selector.py). Every other preset
    # keeps render_notes.py's own defaults (exact harmonics, single voice).
    if "inharmonicity" in render_args:
        command += ["--inharmonicity", str(render_args["inharmonicity"])]
    if "unison_voices" in render_args:
        command += ["--unison_voices", str(render_args["unison_voices"])]
    if "unison_detune_cents" in render_args:
        command += ["--unison_detune_cents", str(render_args["unison_detune_cents"])]

    command += ["--midi", *[str(note) for note in midi_notes]]

    run_command(command)


def postprocess_if_needed(mode_config: dict):
    post = mode_config.get("postprocess", {})

    if not post.get("bass_click", False):
        return

    check_file(ENHANCE_SCRIPT, "Cannot find enhance_rendered_notes.py.")

    amount = post.get("click_amount", 0.45)
    output_wav = TIMBRE_DIR / "output.wav"

    command = [
        sys.executable,
        str(ENHANCE_SCRIPT),
        "--folder",
        str(RENDERED_DIR),
        "--amount",
        str(amount),
        "--include_output",
        str(output_wav),
    ]

    run_command(command)


def save_summary(
    predicted_class: str,
    predicted_family: str,
    classifier_output: str,
    mode_config: dict,
):
    shutil.copy2(INPUT_AUDIO, LATEST_RUN / "input_target.wav")

    with open(LATEST_RUN / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("Capstone Software System - Run Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"input_audio: {INPUT_AUDIO}\n")
        f.write(f"predicted_class: {predicted_class}\n")
        f.write(f"predicted_family: {predicted_family}\n")
        f.write(f"selected_mode: {mode_config['mode_name']}\n")
        f.write(f"render_midi_notes_count: {len(mode_config['midi_notes'])}\n")
        f.write(f"render_midi_notes: {mode_config['midi_notes']}\n")
        f.write(f"train_args: {mode_config['train_args']}\n")
        f.write(f"render_args: {mode_config['render_args']}\n")
        f.write(f"postprocess: {mode_config.get('postprocess', {})}\n\n")

        f.write("classifier_output:\n")
        f.write(classifier_output)
        f.write("\n")

        f.write("outputs:\n")
        f.write(f"- timbre_learning: {TIMBRE_DIR}\n")
        f.write(f"- rendered_notes: {RENDERED_DIR}\n")
        f.write(f"- learned_params: {LEARNED_PARAMS}\n")


def main():
    print("=" * 80)
    print("Capstone Software System")
    print("36-note instrument-specific generation modes enabled")
    print("=" * 80)

    check_file(INPUT_AUDIO, "Put a single-note audio file at input/target.wav.")
    check_file(PREDICT_SCRIPT, "Cannot find predict_instrument.py.")
    check_file(TRAIN_SCRIPT, "Cannot find train_timbre.py.")
    check_file(RENDER_SCRIPT, "Cannot find render_notes.py.")

    prepare_output_folder()

    predicted_class, predicted_family, classifier_output = predict_instrument()
    # midi_notes here is provisional -- this recording's own pitch isn't
    # known until train_timbre runs. Re-fetched below with the real f0_hz
    # so the rendered range centers on what was actually learned.
    mode_config = get_mode_config(predicted_class)

    print("\nPrediction / Mode Selection")
    print("-" * 80)
    print(f"Predicted instrument: {predicted_class}")
    print(f"Predicted family: {predicted_family}")
    print(f"Selected mode: {mode_config['mode_name']}")
    print(f"Train args: {mode_config['train_args']}")
    print(f"Render args: {mode_config['render_args']}")
    print(f"Postprocess: {mode_config.get('postprocess', {})}")

    train_timbre(mode_config)

    source_f0_hz = None
    if LEARNED_PARAMS.exists():
        try:
            with open(LEARNED_PARAMS, "r", encoding="utf-8") as f:
                source_f0_hz = json.load(f).get("f0_hz")
        except (OSError, json.JSONDecodeError):
            source_f0_hz = None
    mode_config = get_mode_config(predicted_class, source_f0_hz=source_f0_hz)

    print(f"\nSource pitch: {source_f0_hz} Hz" if source_f0_hz else "\nSource pitch: unknown, using class-wide range")
    print(f"Number of MIDI notes: {len(mode_config['midi_notes'])}")
    print(f"MIDI notes: {mode_config['midi_notes']}")

    render_notes(mode_config)
    postprocess_if_needed(mode_config)
    save_summary(predicted_class, predicted_family, classifier_output, mode_config)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Final output folder: {LATEST_RUN}")
    print(f"Rendered notes folder: {RENDERED_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
