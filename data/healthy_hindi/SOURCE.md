# Hindi calibration source data

`models/hindi_calibration.json` was generated from 15 clips sampled from the
**OpenSLR SLR103** "Multilingual and code-switching ASR challenge" Hindi test
set (Microsoft Research Open Data), downloaded 2026-09-08:

- Source: https://www.openslr.org/103/ (`Hindi_test.tar.gz`)
- License: Microsoft Research Open Data license (see the license link on the
  OpenSLR page) — permits computational/research use; the raw audio is not
  redistributed from this repo (see `.gitignore`), only the derived per-feature
  mean offsets.
- Clips used: `2131_098`, `1119_029`, `5945_089`, `3701_029`, `0865_086`,
  `5532_088`, `4792_066`, `2415_086`, `5169_029`, `4663_003`, `1139_028`,
  `2658_087`, `0330_003`, `3226_003`, `0821_086` (15 distinct speakers),
  converted to mono 16kHz 16-bit PCM.

**Important limitation**: this is general adult Hindi speech read for an ASR
benchmark, not a set of age-matched or clinically-screened "cognitively
healthy" controls the way the original 7 English healthy samples were intended
to be. It's a reasonable proxy for calibrating the ASR-derived linguistic
features (word segmentation, pause detection) across languages, but it isn't
equivalent to a real Hindi clinical control group. Treat `hindi_calibration.json`
accordingly, and replace it with real Hindi speaker recordings if you get
access to any — just add `.wav` files to this folder and re-run
`python calibrate_hindi.py` from `src/`.
