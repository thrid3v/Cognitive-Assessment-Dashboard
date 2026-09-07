import os
import wave
import json
import pickle
import numpy as np
import pandas as pd
import librosa
from vosk import Model, KaldiRecognizer

# --- DYNAMIC PATH SETUP (THE FIX) ---
# Get the directory where THIS file (predictor.py) is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (neuro-sentinel/)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Construct absolute paths that work on both Windows and Cloud
VOSK_EN_PATH = os.path.join(ROOT_DIR, "models", "vosk-model-small-en-us-0.15")
VOSK_HI_PATH = os.path.join(ROOT_DIR, "models", "vosk-model-small-hi-0.22")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "neuro_model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")
DATASET_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
HINDI_CALIBRATION_PATH = os.path.join(ROOT_DIR, "models", "hindi_calibration.json")

# The 5 biomarkers derived from ASR word timestamps (language-dependent).
# The other 4 (acoustic_texture, mfcc_delta, speech_brightness, emotional_range)
# come straight from Librosa on the raw waveform and are language-independent,
# so they're never touched by Hindi calibration.
LINGUISTIC_FEATURES = ["pause_rate", "vocab_richness", "word_count", "speech_rate", "initial_latency"]

# --- 1. LOAD MODELS ---
print(f"🔎 Looking for models in: {os.path.join(ROOT_DIR, 'models')}")

clf = None
scaler = None
vosk_model_en = None
vosk_model_hi = None

# Load Vosk (English)
if not os.path.exists(VOSK_EN_PATH):
    # Fallback: Sometimes Cloud puts models in src/models
    VOSK_EN_PATH = os.path.join(CURRENT_DIR, "models", "vosk-model-small-en-us-0.15")

if os.path.exists(VOSK_EN_PATH):
    try:
        vosk_model_en = Model(VOSK_EN_PATH)
        print("✅ Vosk EN Model Loaded!")
    except Exception as e:
        print(f"❌ Vosk EN Load Error: {e}")
else:
    print(f"⚠️ CRITICAL: Vosk EN model not found at {VOSK_EN_PATH}")

# Load Vosk (Hindi) — optional; Hindi mode degrades gracefully if absent
if not os.path.exists(VOSK_HI_PATH):
    VOSK_HI_PATH = os.path.join(CURRENT_DIR, "models", "vosk-model-small-hi-0.22")

if os.path.exists(VOSK_HI_PATH):
    try:
        vosk_model_hi = Model(VOSK_HI_PATH)
        print("✅ Vosk HI Model Loaded!")
    except Exception as e:
        print(f"❌ Vosk HI Load Error: {e}")
else:
    print(f"ℹ️ Vosk HI model not found at {VOSK_HI_PATH} — Hindi mode unavailable until it is downloaded.")


def _get_vosk_model(language="en"):
    return vosk_model_hi if language == "hi" else vosk_model_en


def is_hindi_available():
    return vosk_model_hi is not None


def _load_hindi_calibration():
    if not os.path.exists(HINDI_CALIBRATION_PATH):
        print(f"ℹ️ No Hindi calibration found at {HINDI_CALIBRATION_PATH} — using raw cross-lingual transfer.")
        return {}
    try:
        with open(HINDI_CALIBRATION_PATH, 'r') as f:
            offsets = json.load(f)
        print(f"✅ Hindi calibration loaded: {offsets}")
        return offsets
    except Exception as e:
        print(f"⚠️ Hindi calibration load error: {e} — falling back to raw cross-lingual transfer.")
        return {}


hindi_calibration = _load_hindi_calibration()

# Load Classifier & Scaler
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            clf = pickle.load(f)
            print("✅ Classifier Loaded!")
    else:
        print(f"⚠️ Classifier not found at {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
            print("✅ Scaler Loaded!")
except Exception as e:
    print(f"⚠️ Model/Scaler Load Error: {e}")


def generate_clinical_summary(features_df):
    """Generates softer, elderly-friendly clinical observations."""
    try:
        if features_df is None or features_df.empty:
            return "No data available for summary."

        # Extract values safely
        p_rate = features_df['pause_rate'].iloc[0]
        s_rate = features_df.get('speech_rate', pd.Series([0])).iloc[0]
        
        observations = []
        # Using supportive, clinical language
        if p_rate > 0.4:
            observations.append("Observed slight hesitations in natural speech rhythm.")
        if s_rate < 1.8 and s_rate > 0:
            observations.append("Speech tempo is currently below the standardized baseline.")
            
        if not observations:
            return "Speech biomarkers reflect a stable and healthy cognitive baseline."
        
        return " | ".join(observations)
    except Exception as e:
        return f"Summary generation error: {str(e)}"

def extract_features_from_audio(file_path, language="en"):
    """ Extracts all biomarkers using Librosa and Vosk """
    try:
        # 1. ACOUSTIC ANALYSIS (Librosa)
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc) # Global mean
        mfcc_delta = np.mean(librosa.feature.delta(mfcc))
        
        # Spectral Centroid
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Pitch / Emotional Range
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        emotional_range = np.var(pitch_values) if len(pitch_values) > 0 else 0
        
        # 2. LINGUISTIC ANALYSIS (Vosk)
        model_to_use = _get_vosk_model(language)
        if model_to_use is None:
            print(f"⚠️ No Vosk model available for language='{language}'.")
            return None

        wf = wave.open(file_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio format mismatch. Expecting Mono 16kHz WAV.")
            # return None # You might want to handle conversion here if needed

        rec = KaldiRecognizer(model_to_use, wf.getframerate())
        rec.SetWords(True)
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0: break
            if rec.AcceptWaveform(data):
                part = json.loads(rec.Result())
                if 'result' in part: results.extend(part['result'])
        
        final_part = json.loads(rec.FinalResult())
        if 'result' in final_part: results.extend(final_part['result'])
        wf.close()
        
        if not results: 
            print("No words detected by Vosk.")
            return None

        # Calculate Metrics
        initial_latency = results[0]['start']
        total_duration = results[-1]['end'] - results[0]['start']
        
        pause_time = 0.0
        for i in range(len(results) - 1):
            gap = results[i+1]['start'] - results[i]['end']
            if gap > 0.25:
                pause_time += gap
        
        pause_rate = pause_time / total_duration if total_duration > 0 else 0
        words = [w['word'] for w in results]
        ttr = len(set(words)) / len(words) if words else 0
        speech_rate = len(words) / total_duration if total_duration > 0 else 0
        
        # RETURN DATAFRAME (Must match model columns exactly)
        return pd.DataFrame([{
            "pause_rate": round(pause_rate, 3),
            "vocab_richness": round(ttr, 3),
            "word_count": len(words),
            "initial_latency": round(initial_latency, 3),
            "acoustic_texture": round(mfcc_mean, 3),
            "speech_brightness": round(spec_centroid, 3),
            "mfcc_delta": round(mfcc_delta, 4),
            "emotional_range": round(emotional_range, 3),
            "speech_rate": round(speech_rate, 3)
        }])

    except Exception as e:
        print(f"❌ Feature Extraction Error: {e}")
        return None

def apply_hindi_calibration(features_df):
    """Shifts the 5 ASR-derived features by the English-vs-Hindi healthy-baseline
    offset (see calibrate_hindi.py). No-op until that calibration file exists,
    which means Hindi runs in raw cross-lingual-transfer mode until then."""
    if not hindi_calibration:
        return features_df
    calibrated = features_df.copy()
    for feature in LINGUISTIC_FEATURES:
        if feature not in calibrated.columns or feature not in hindi_calibration:
            continue
        new_val = max(0.0, calibrated.at[0, feature] + hindi_calibration[feature])
        if feature == "word_count":
            new_val = round(new_val)
        calibrated.at[0, feature] = new_val
    return calibrated


def get_prediction(audio_path, language="en"):
    """ Returns: (Label, Confidence, Features) """
    if clf is None:
        return "Model Error", 0.0, None

    features_df = extract_features_from_audio(audio_path, language=language)

    if features_df is None:
        return "Processing Error", 0.0, None

    if language == "hi":
        features_df = apply_hindi_calibration(features_df)

    # Scale features if scaler exists
    features_final = features_df
    if scaler is not None:
        try:
            features_final = scaler.transform(features_df)
        except Exception as e:
            print(f"Scaling warning: {e}")

    try:
        # Get Probabilities
        probabilities = clf.predict_proba(features_final)[0]
        confidence = float(np.max(probabilities))
        
        # Get Class
        prediction_class = clf.predict(features_final)[0]
        
        labels = {0: "Healthy", 1: "Mild Decline", 2: "Moderate Decline", 3: "Severe Decline"}
        result_text = labels.get(prediction_class, "Unknown")
        
        return result_text, confidence, features_df
    
    except Exception as e:
        print(f"Prediction logic error: {e}")
        return "Error", 0.0, features_df


def calculate_current_accuracy():
    """Calculates accuracy against local dataset if available."""
    try:
        if not os.path.exists(DATASET_PATH) or clf is None:
            return 0.0
            
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Check if dataset has required columns
        required_cols = ['label'] # Add feature columns check if needed
        if not all(col in df.columns for col in required_cols):
            return 0.0

        X = df.drop(['label', 'filename'], axis=1, errors='ignore')
        y = df['label']
        
        # Scale
        if scaler:
            X = scaler.transform(X)
        
        # Predict
        from sklearn.metrics import accuracy_score
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        
        return round(acc * 100, 2)
    except Exception as e:
        print(f"Accuracy calc error: {e}")
        return 0.0
