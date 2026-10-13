# NeuroSentinel - Cognitive Assessment Dashboard

![Version](https://img.shields.io/badge/version-1.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

**NeuroSentinel** is an AI-powered cognitive assessment system that analyzes speech patterns to detect early signs of cognitive decline. The system extracts 9 neurological biomarkers from voice recordings and uses machine learning to classify cognitive health status.

---

**STREAMLIT DEMO LIVE- https://devoops-eq2x4aewddar6wxa3qmwk6.streamlit.app/**    

## 🚀 Quick Start - How to Run

### Prerequisites
- Python 3.8 or higher
- Microphone access (for live recording)
- ~500MB disk space (for Vosk speech recognition model)

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd hack_sphere
```

2. **Install system dependencies** (required for audio processing)

**macOS:**
```bash
brew install ffmpeg libsndfile
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1
```

**Windows:**
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Or use Chocolatey: `choco install ffmpeg`

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch the Application**
```bash
cd src
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

> **Note**: The Vosk speech recognition model, trained ML model, and scaler are already included in the repository under the `models/` directory.

### Optional: Retrain the Model

The model is trained on real clinical speech recordings (Cookie Theft picture-description task), not on data shipped in this repo — the source audio is a third-party research dataset and isn't redistributed here. To retrain with that data (or your own):

1. Edit `EXTERNAL_CSV_PATH` and `EXTERNAL_AUDIO_ROOT` at the top of `src/extract_features.py` to point at your labeled CSV (needs `Record-ID`, `Class` columns, with `Class` in `HC`/`MCI`/`Dementia`) and the matching audio folder (`<root>/<Record-ID>/<Record-ID>__CTD.wav`).
2. Run the pipeline from the project root:

```bash
python src/extract_features.py   # -> data/dataset.csv (biomarkers only, no raw audio/PII)
python src/train_model.py        # -> models/neuro_model.pkl, models/scaler.pkl
```

`data/dataset.csv` holds only the 9 numeric biomarkers + label per recording — safe to commit even when the source audio isn't.

---

## 📊 9-Point Neurological Biomarker Panel

NeuroSentinel analyzes speech using **9 distinct biomarkers** across acoustic, prosodic, and linguistic dimensions:

### 1. **Pause Rate** 
- **Type**: Linguistic Timing
- **Description**: Ratio of pause time to total speech duration
- **Calculation**: `pause_time / total_duration` (pauses > 0.25s)
- **Clinical Significance**: Elevated pause rates (>0.35) indicate hesitations and word-finding difficulties, common in early cognitive decline
- **Healthy Range**: < 0.35 (mean − 1 std of the real Healthy class, `data/dataset.csv`)

### 2. **Vocabulary Richness (Type-Token Ratio)**
- **Type**: Linguistic Complexity
- **Description**: Ratio of unique words to total words spoken
- **Calculation**: `len(unique_words) / len(total_words)`
- **Clinical Significance**: Reduced lexical diversity suggests semantic memory impairment
- **Healthy Range**: > 0.55 (mean − 1 std of the real Healthy class)

### 3. **Word Count**
- **Type**: Linguistic Fluency
- **Description**: Total number of words spoken in the recording
- **Calculation**: Direct count from speech-to-text transcription
- **Clinical Significance**: Lower word counts may indicate reduced verbal fluency
- **Healthy Range**: > 80 words (for an unscripted ~1-2 minute description task; scales with recording length)

### 4. **Speech Rate**
- **Type**: Temporal Dynamics
- **Description**: Words spoken per second
- **Calculation**: `word_count / total_duration`
- **Clinical Significance**: Slowed speech rate (<1.9 wps) correlates with processing speed deficits
- **Healthy Range**: > 1.9 words per second

### 5. **Initial Latency**
- **Type**: Response Time
- **Description**: Time delay before first word is spoken
- **Calculation**: Timestamp of first word from recording start
- **Clinical Significance**: Delayed response indicates cognitive processing delays
- **Healthy Range**: < 3.0 seconds (real unscripted responses to a picture-description prompt vary a lot more than a scripted 0.5s cutoff assumed)

### 6. **Acoustic Texture (MFCC Mean)**
- **Type**: Acoustic Feature
- **Description**: Mean across all 13 Mel-Frequency Cepstral Coefficients
- **Calculation**: `np.mean(librosa.feature.mfcc(y, sr, n_mfcc=13))`
- **Clinical Significance**: Reflects voice quality and articulatory precision; changes may indicate motor speech disorders
- **Healthy Range**: > -30 (mean − 1 std of the real Healthy class; this range changed substantially from older versions of this doc because the calculation itself changed from "mean of 1st MFCC" to "mean of all 13 MFCCs" — don't reuse the old -350 cutoff)

### 7. **MFCC Delta**
- **Type**: Acoustic Dynamics
- **Description**: Rate of change in MFCCs (spectral dynamics)
- **Calculation**: `np.mean(librosa.feature.delta(mfcc))`
- **Clinical Significance**: High variability suggests unstable articulation; low variability indicates monotonous speech
- **Healthy Range**: Stable values (|delta| < 0.015)

### 8. **Speech Brightness (Spectral Centroid)**
- **Type**: Acoustic Feature
- **Description**: Center of mass of the frequency spectrum
- **Calculation**: `np.mean(librosa.feature.spectral_centroid(y, sr))`
- **Clinical Significance**: Lower values indicate muffled or less energetic speech, associated with reduced vocal effort
- **Healthy Range**: > 1600 Hz

### 9. **Emotional Range (Pitch Variance)**
- **Type**: Prosodic Feature
- **Description**: Variance in fundamental frequency (F0)
- **Calculation**: `np.var(pitch_values)` from `librosa.piptrack()`
- **Clinical Significance**: Reduced pitch variance indicates flat affect and emotional blunting
- **Healthy Range**: > 1,000,000 (high expressiveness)

---

## 🧠 Machine Learning Models

### Primary Classification Model
- **Algorithm**: Random Forest Classifier, binary output (Healthy / Dementia — see "Classification Categories" below for why there's no middle label)
- **Implementation**: `sklearn.ensemble.RandomForestClassifier`
- **Training Data**: 157 real clinical recordings (Cookie Theft picture-description task). Source labels are Healthy / MCI / Dementia by clinicians, but MCI is folded into the Healthy bucket for training (82 Healthy + 59 MCI = 141 "Healthy" vs. 16 Dementia) — see "Classification Categories" below for why
- **Feature Preprocessing**: StandardScaler normalization
- **Class Imbalance Handling**: `class_weight` grid-searched; decision uses a probability threshold (not bare most-likely-class) tuned for conservative flagging — see "Classification Categories"
- **Hyperparameter Optimization**: GridSearchCV with 5-fold cross-validation

#### Optimized Hyperparameters
```python
{
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}
```

### Speech Recognition Model
- **Model**: Vosk (Kaldi-based ASR)
- **Version**: vosk-model-small-en-us-0.15 (English), vosk-model-small-hi-0.22 (Hindi)
- **Purpose**: Speech-to-text transcription for linguistic feature extraction
- **Language**: English (US) primary; Hindi supported via cross-lingual transfer (see below)
- **Size**: ~40MB (EN) + ~45MB (HI)

### Hindi Language Support (Experimental)

No labeled Hindi cognitive-decline speech dataset exists publicly or was otherwise accessible for this project, so Hindi is supported via cross-lingual transfer: a Hindi recording is transcribed with `vosk-model-small-hi-0.22`, all 9 biomarkers are computed, but **only the 4 waveform-derived biomarkers** (acoustic texture, MFCC delta, speech brightness, emotional range) are scored — by `models/neuro_model_acoustic.pkl`, a separate Random Forest trained on the same 157 English recordings using just those 4 features. The 5 speech-to-text-derived biomarkers (pause rate, vocabulary richness, word count, speech rate, initial latency) are still shown in the UI for information, but don't feed the classification decision.

**Why not use all 9 features, or calibrate the other 5?** Both were tried and rejected on evidence, not just caution:
- The 5 ASR-derived biomarkers depend on language-specific transcription (word segmentation, what counts as a pause) and the only Hindi reference audio available to calibrate them (15 clips from the OpenSLR SLR103 ASR benchmark) turned out to be 5-second scripted sentence reads, not spontaneous narration like the live app's task — their pause rate and word count differ from the English baseline by 12-16x, almost entirely due to utterance length, not language. Calibrating against that reference would mostly correct for "short scripted sentence vs. long spontaneous narrative" and badly distort real recordings.
- The acoustic-only model performs about as well as the full 9-feature model on English data (sensitivity 25% vs. 31%, specificity 97% vs. 87% at the same threshold — see `models/metrics.json`), so dropping the 5 ASR-derived features costs little and is actually more conservative.
- However, testing the acoustic-only model against those same 15 Hindi reference clips classified **0 of 15 as Healthy** - not because cross-lingual transfer failed, but because the clips are genuine 8kHz telephone-bandwidth recordings (confirmed: 0.08% of spectral energy above 4kHz, vs 7% for the English recordings) completely lacking the frequency content `speech_brightness`/MFCCs are computed from. That's a recording-equipment artifact, not a language or health signal — but it meant this particular reference set couldn't validate (or invalidate) the approach either way.

**Bottom line**: the acoustic-only architecture is the more defensible design (fewer features that can plausibly be confounded by language), but it is **not empirically validated on any real Hindi speech** — favorable or not. If you get access to even a handful of Hindi speakers recorded through this app itself (same mic, same spontaneous-description task as the live assessment), that would be real validation data; a mismatched reference corpus (different recording equipment and/or a different speech task) isn't.

**Setup**: download and extract [`vosk-model-small-hi-0.22`](https://alphacephei.com/vosk/models) into `models/vosk-model-small-hi-0.22/`. No additional setup is needed — Hindi mode activates automatically once that model is present.

### Feature Extraction Libraries
- **Librosa**: Acoustic and prosodic feature extraction (MFCCs, spectral centroid, pitch tracking)
- **Vosk**: Real-time speech recognition with word-level timestamps
- **NumPy**: Statistical computations

---

## 🎯 Classification Categories

The system outputs **only two labels: Healthy or Dementia.** There is deliberately no "MCI"/"mild" middle label in the app, for two reasons:

1. **Early testing showed normal conversational pauses/hesitation getting flagged as decline.** A 3-class model (Healthy / MCI / Dementia) was too willing to call ordinary variation "MCI."
2. **The decision is conservative by design**: "Dementia" is only reported when the model's predicted probability clears a threshold (currently **0.40**), not just whichever class is more likely. Below that threshold, the result is "Healthy" — full stop, no intermediate label.

MCI recordings are folded into the "Healthy" training label (not into "Dementia") specifically so the model learns to leave ambiguous/mild cases alone rather than flag them — see `train_model.py`.

**The real trade-off** (measured via repeated cross-validation, not a single lucky/unlucky split — see `models/metrics.json`):

| Threshold | Sensitivity (catches real Dementia) | Specificity (leaves Healthy/MCI alone) |
|---|---|---|
| 0.30 | 44% | 75% |
| **0.40 (current)** | **31%** | **87%** |
| 0.50 | 12% | 93% |
| 0.60+ | ~0% | ~99% |

With only 16 real Dementia examples in the training data, there is no threshold that is both conservative (few false alarms) and sensitive (catches most real cases) — this is a genuine limitation of the data, not a tunable bug. At 0.40, the model catches roughly **1 in 3** real Dementia cases and leaves **87%** of Healthy/MCI speakers alone. **This is a screening aid that errs toward not alarming healthy users, not a tool to rule out dementia** — a "Healthy" result does not mean decline has been ruled out.

**Recording quality notes**: when the result is "Healthy," the app also checks the recording itself (duration, word count, volume, speaking pace, pause pattern) and surfaces short tips like *"Recording was quite short — try speaking for 30-60 seconds"* when something about the recording — not necessarily the person's cognition — looks like it could have affected the result. See `predictor.assess_recording_quality()`.

---

## 📁 Project Structure

```
hack_sphere/
├── src/
│   ├── app.py                      # Main Streamlit application
│   ├── predictor.py                # Prediction engine and feature extraction
│   ├── extract_features.py         # Batch feature extraction from labeled audio (external dataset)
│   ├── train_model.py              # Model training with hyperparameter tuning
│   ├── generate_synthetic_data.py  # (legacy, unused by default) Gaussian data augmentation
│   ├── db.py                       # SQLite database operations
│   ├── report.py                   # PDF report generation
│   └── visualize_results.py        # Confusion matrix visualization
├── models/
│   ├── neuro_model.pkl             # Trained Random Forest model (9 features, English)
│   ├── scaler.pkl                  # Feature scaler (StandardScaler)
│   ├── neuro_model_acoustic.pkl    # Acoustic-only model (4 features, used for Hindi)
│   ├── scaler_acoustic.pkl         # Feature scaler for the acoustic-only model
│   ├── metrics.json                # Honest held-out/cross-validated accuracy (not training accuracy)
│   ├── vosk-model-small-en-us-0.15/  # English speech recognition model
│   └── vosk-model-small-hi-0.22/     # Hindi speech recognition model (optional)
├── data/
│   └── dataset.csv                 # Extracted biomarkers + labels (no raw audio/PII)
├── neuro.db                        # SQLite database for patient records
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🔬 Technical Workflow

### 1. **Audio Recording**
- User records voice sample via Streamlit audio input
- Audio saved as WAV file (16kHz, mono)

### 2. **Feature Extraction**
```python
# Acoustic features (Librosa)
- MFCC analysis
- Spectral centroid calculation
- Pitch tracking

# Linguistic features (Vosk)
- Speech-to-text transcription
- Word-level timestamps
- Pause detection
```

### 3. **Preprocessing**
- Feature scaling using saved `StandardScaler`
- Normalization ensures balanced feature importance

### 4. **Prediction**
- Random Forest classifier predicts cognitive status
- Confidence score calculated from class probabilities
- Results stored in SQLite database

### 5. **Visualization**
- Radar chart comparing patient data to healthy baseline
- Historical trend analysis (pause rate, word count, speech rate)
- PDF report generation with clinical interpretation

---

## 📈 Model Performance

**Accuracy is not reported anywhere in this app, deliberately.** On a binary problem that's 141 Healthy vs. 16 Dementia (~90%/10%), a model that always says "Healthy" scores ~90% accuracy while being useless — so "accuracy" would be actively misleading here, the same way the old training-accuracy badge was (see git history). The metrics that matter are **sensitivity** (fraction of real Dementia cases caught) and **specificity** (fraction of Healthy/MCI people correctly left alone), computed via **repeated cross-validation** (5-fold × 20 repeats = 100 out-of-fold evaluations per sample, not a single train/test split, which would be too noisy with only 16 positive examples):

- **Full 9-feature model** (English): sensitivity 31%, specificity 87% at the 0.40 decision threshold — see `models/metrics.json` for the exact counts (true/false positives/negatives).
- **Acoustic-only model** (Hindi): sensitivity 25%, specificity 97% — fewer features, so it's naturally more conservative.

Treat this as a research-grade screening aid tuned to minimize false alarms on ordinary speech variation, not a diagnostic tool and not something that rules out dementia when it says "Healthy." See "Classification Categories" above for the full sensitivity/specificity trade-off at other thresholds and why 0.40 was chosen.

- **Optimization Techniques**:
  - Feature scaling (StandardScaler)
  - Class-weight balancing (grid-searched)
  - Hyperparameter tuning via GridSearchCV, scored on `balanced_accuracy` (plain accuracy would reward configs that just lean on the majority Healthy class)
  - Decision threshold tuned on out-of-fold cross-validated probabilities, not training-set predictions

---

## 🗄️ Database Schema

**Table**: `records`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key (auto-increment) |
| timestamp | TEXT | Assessment date/time |
| patient_id | TEXT | Patient identifier |
| risk_label | TEXT | Classification result |
| confidence | REAL | Prediction confidence (%) |
| pause_rate | REAL | Biomarker value |
| vocab_richness | REAL | Biomarker value |
| word_count | INTEGER | Biomarker value |
| speech_rate | REAL | Biomarker value |
| initial_latency | REAL | Biomarker value |
| acoustic_texture | REAL | Biomarker value |
| mfcc_delta | REAL | Biomarker value |
| speech_brightness | REAL | Biomarker value |
| emotional_range | REAL | Biomarker value |

---

## 🎨 User Interface Features

### Tab 1: New Assessment
- Circular red recording button with live waveform
- One-click analysis with real-time biomarker extraction
- Audio playback for verification

### Tab 2: Biomarker Details
- 9-metric dashboard with color-coded indicators
- Neurological fingerprint radar chart
- Clinical observations in elderly-friendly format
- Recording quality notes (e.g. "recording was short" / "audio volume seemed low") when the recording itself may have affected the result — not a diagnostic claim
- PDF report download
- Manual trial tracking (confirm correct/incorrect)

### Tab 3: Clinical History
- Longitudinal trend graphs for key biomarkers
- Historical data table for patient records
- Risk threshold indicators

---

## 🔧 Dependencies

```txt
streamlit          # Web application framework
pandas             # Data manipulation
plotly             # Interactive visualizations
numpy              # Numerical computations
librosa            # Audio feature extraction
vosk               # Speech recognition
scikit-learn       # Machine learning
fpdf               # PDF report generation
```

---

## 🧪 Development Commands

### Extract Features from Audio Files
```bash
python src/extract_features.py
```
Reads `EXTERNAL_CSV_PATH`/`EXTERNAL_AUDIO_ROOT` (edit these constants in the script to point at your labeled dataset) and writes `data/dataset.csv`.

### Train Model
```bash
python src/train_model.py
```
Performs hyperparameter tuning on `data/dataset.csv` and saves the optimized model to `models/neuro_model.pkl`.

### Run Application
```bash
streamlit run app.py
```

---

## 📄 PDF Report Generation

Each assessment generates a clinical report containing:
- Patient ID and timestamp
- Risk assessment (color-coded)
- Complete biomarker table with reference ranges
- Clinical interpretation based on findings
- Recommendations for follow-up

---

## 🔒 Privacy & Security

- All processing occurs **locally** (edge AI)
- No data transmitted to external servers
- SQLite database stored locally
- HIPAA-compliant architecture (local deployment)

---

## 🎯 Clinical Use Cases

1. **Early Screening**: Detect subtle cognitive changes before clinical symptoms
2. **Longitudinal Monitoring**: Track cognitive trajectory over time
3. **Treatment Response**: Measure intervention effectiveness
4. **Research**: Collect standardized speech biomarkers for studies

---

## ⚠️ Limitations & Disclaimers

- **Not a diagnostic tool**: For research and screening purposes only
- **Requires clinical validation**: Predictions should be confirmed by healthcare professionals
- **Language limitation**: Native support for English (US); Hindi support uses a separate acoustic-only model and is completely unvalidated on real Hindi speech, in either direction (see Hindi Language Support above)
- **Audio quality**: Requires clear recordings in quiet environments
- **Sample size**: Trained on 157 real recordings (141 Healthy incl. MCI / 16 Dementia) — with only 16 positive examples, sensitivity/specificity estimates have real uncertainty, and there's no threshold that's both conservative and reliably catches most real cases (see "Classification Categories")
- **Single task**: Trained only on the Cookie Theft picture-description task; may not generalize to other speech elicitation tasks

---

## 🤝 Contributing

To extend the system:
1. Add new labeled recordings to your external dataset (CSV + audio folder)
2. Run `extract_features.py` to update `data/dataset.csv`
3. Retrain the model using `train_model.py`

---

## 📚 References

### Biomarker Research
- Pause rate and cognitive decline: König et al. (2015)
- Lexical diversity in dementia: Fraser et al. (2016)
- Acoustic features in neurodegeneration: Skodda et al. (2011)

### Technologies
- [Librosa Documentation](https://librosa.org/)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)

---

## 📞 Support

For technical issues or questions:
- **Missing audio libraries**: Ensure `ffmpeg` and `libsndfile` are installed (see Installation Steps)
- **Import errors**: Run `pip install -r requirements.txt` to install all Python dependencies
- **Audio format issues**: Check that files are WAV, 16kHz, mono format
- **Model not found**: Verify the Vosk model exists in `models/vosk-model-small-en-us-0.15/`
- **Runtime errors**: Review terminal logs for detailed error messages

### Common Issues

**"No module named 'librosa'" or audio loading fails:**
```bash
# Install system dependencies first
brew install ffmpeg libsndfile  # macOS
# Then reinstall librosa
pip install --upgrade librosa
```

**"Model not found" error:**
- Ensure you're in the `src/` directory when running the app
- Verify `models/vosk-model-small-en-us-0.15/` exists in the parent directory

---

## 📝 License

MIT License - See LICENSE file for details

---

**Version**: 1.2 (Clinical Research Build)  
**Last Updated**: 2026-02-08  
**System**: NeuroSentinel Edge AI


:0
