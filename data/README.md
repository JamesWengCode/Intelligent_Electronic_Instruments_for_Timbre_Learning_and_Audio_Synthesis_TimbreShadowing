# Training Data

`data/` is not tracked in this repo (~2 GB of audio). This describes the
folder layout the classifier (`src/train_classifier.py`) expects, and where
the original training audio came from, so it can be rebuilt.

## Layout

```
data/<family>/<instrument>/*.wav (or .aif/.aiff)
```

Family and instrument folder names must match
`src/instrument_taxonomy.json`:

| Family | Instruments |
|---|---|
| strings | violin, viola, cello, double_bass |
| woodwinds | flute, oboe, clarinet, saxophone, bassoon |
| brass | trumpet, trombone, french_horn, tuba |
| keyboard | acoustic_piano, electric_piano, organ, synth |
| guitar | acoustic_guitar, electric_guitar, electric_bass |

## Source

| Source | Files | Classes |
|---|---:|---|
| [University of Iowa Musical Instrument Samples](http://theremin.music.uiowa.edu/MIS.html) (`Instrument.style.dynamic[.string].pitch.stereo.aif`) | 992 | violin, viola, cello, double_bass, flute, oboe, clarinet, saxophone, bassoon, trumpet, trombone, french_horn, tuba |
| [TinySOL](https://zenodo.org/records/3685367) (`Abbrev-ord-pitch-dynamic-...wav`, e.g. `Hn-ord-A#1-mf-N-T17d.wav`) | 1,915 | same as above except double_bass |
| [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) (`*_electronic_*.wav`) | 2,819 | acoustic_piano, synth, electric_guitar, electric_bass |

The 12 TinySOL recordings used as an evaluation set elsewhere in the
project are at different pitches than the TinySOL material used here for
training -- the two sets don't overlap.

## Format Expectations

`train_classifier.py` pads/trims every clip to a fixed duration
(`DURATION_SECONDS` in that file). Keep each class's clip lengths
consistent (all close to that duration) -- a clip much shorter than the
rest of its class gets mostly zero-padded, which pushes it out of
distribution and can cause confident misclassification at inference time.
