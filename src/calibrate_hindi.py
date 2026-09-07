import os
import json
import pandas as pd
from predictor import extract_features_from_audio, LINGUISTIC_FEATURES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
HEALTHY_HINDI_DIR = os.path.join(ROOT_DIR, "data", "healthy_hindi")
DATASET_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
OUTPUT_PATH = os.path.join(ROOT_DIR, "models", "hindi_calibration.json")


def compute_hindi_healthy_means():
    if not os.path.exists(HEALTHY_HINDI_DIR):
        print(f"⚠️ {HEALTHY_HINDI_DIR} does not exist.")
        return None

    wav_files = [f for f in os.listdir(HEALTHY_HINDI_DIR) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"⚠️ No .wav files found in {HEALTHY_HINDI_DIR}.")
        print("   Add a few healthy Hindi recordings (mono, 16kHz, 16-bit WAV) and re-run this script.")
        return None

    rows = []
    for fname in wav_files:
        feats = extract_features_from_audio(os.path.join(HEALTHY_HINDI_DIR, fname), language="hi")
        if feats is None:
            print(f"   ❌ {fname} FAILED (skipped).")
            continue
        rows.append(feats)
        print(f"   ✅ {fname} processed.")

    if not rows:
        print("🛑 No usable Hindi recordings were processed. Aborting.")
        return None

    combined = pd.concat(rows, ignore_index=True)
    print(f"📊 Computed Hindi healthy baseline from {len(combined)} recording(s).")
    return combined[LINGUISTIC_FEATURES].mean()


def compute_english_healthy_means():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ {DATASET_PATH} not found.")
        return None

    df = pd.read_csv(DATASET_PATH)
    healthy = df[df['label'] == 0]
    if healthy.empty:
        print("❌ No label==0 (healthy) rows in dataset.csv.")
        return None

    return healthy[LINGUISTIC_FEATURES].mean()


def main():
    print("🔎 Computing Hindi calibration offsets...")
    print(f"📂 Scanning '{HEALTHY_HINDI_DIR}' for healthy Hindi recordings...")

    hindi_means = compute_hindi_healthy_means()
    if hindi_means is None:
        print("🛑 Calibration NOT generated. models/hindi_calibration.json left untouched.")
        return

    english_means = compute_english_healthy_means()
    if english_means is None:
        print("🛑 Calibration NOT generated (missing English baseline).")
        return

    offsets = {f: round(float(english_means[f] - hindi_means[f]), 4) for f in LINGUISTIC_FEATURES}

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(offsets, f, indent=2)

    print(f"✅ Saved calibration offsets to {OUTPUT_PATH}:")
    for k, v in offsets.items():
        print(f"   {k}: {v:+.4f}")


if __name__ == "__main__":
    main()
