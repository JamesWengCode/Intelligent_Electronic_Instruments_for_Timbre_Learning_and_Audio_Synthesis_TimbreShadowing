# Training data

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

An instrument with no data folder is simply skipped during training --
you don't need audio for every instrument in the taxonomy to train the
others.

## Source

The original training set was assembled from three public sample libraries,
identifiable by filename pattern:

- [University of Iowa Musical Instrument Samples](http://theremin.music.uiowa.edu/MIS.html)
  (`Instrument.style.dynamic[.string].pitch.stereo.aif`) -- trombone,
  double_bass, violin, cello, bassoon, clarinet, flute.
- [TinySOL](https://zenodo.org/records/3685367)
  (`Abbrev-ord-pitch-dynamic-...wav`) -- french_horn, trumpet, tuba, viola,
  oboe, saxophone.
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)
  (`*_electronic_*.wav`) -- keyboard/synth, keyboard/acoustic_piano,
  guitar/electric_guitar, guitar/electric_bass.

**Known issue:** `keyboard/acoustic_piano` currently contains NSynth
`keyboard_electronic_*` samples, not real acoustic piano recordings -- the
class label doesn't match the audio's actual timbre. Anyone rebuilding
this dataset should replace it with real acoustic piano samples.

## Format expectations

`train_classifier.py` pads/trims every clip to a fixed duration
(`DURATION_SECONDS` in that file). Keep each class's clip lengths
consistent (all close to that duration) -- a clip much shorter than the
rest of its class gets mostly zero-padded, which pushes it out of
distribution and can cause confident misclassification at inference time.
