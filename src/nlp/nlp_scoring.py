# src/nlp/nlp_scoring.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# =========================
# 1. Load Dataset (train/test)
# =========================
train_df = pd.read_csv("data/processed/train_tone.csv")
test_df = pd.read_csv("data/processed/test_tone.csv")

# =========================
# 2. Load Roberta-GoEmotions Model
# =========================
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()  # inference mode

# =========================
# 3. Ambil 6 label sesuai dataset dair-ai/emotion
# =========================
emotion_order = [
    "sadness", "joy", "love", "anger", "fear", "surprise"
]

# Mapping nama label GoEmotions → index
# Model GoEmotions punya banyak label, kita pilih 6 yang sesuai dataset
label2id = {label: idx for idx, label in enumerate(model.config.id2label.values())}

# =========================
# 4. Fungsi untuk convert text → vector 6 dimensi
# =========================
def get_emotion_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    # Ambil 6 emosi sesuai urutan dataset
    vector_6d = np.array([probs[label2id[e]] for e in emotion_order])

    return vector_6d


# =========================
# 5. Generate X_train, X_test
# =========================
X_train = np.array([get_emotion_vector(t) for t in train_df["text"]])
X_test = np.array([get_emotion_vector(t) for t in test_df["text"]])

# =========================
# 6. Convert tone label → angka kelas
# =========================
tone_classes = {
    "empathetic": 0,
    "friendly": 1,
    "warm": 2,
    "calming": 3,
    "supportive": 4,
    "informative": 5
}

y_train = train_df["tone"].map(tone_classes).values
y_test = test_df["tone"].map(tone_classes).values

# =========================
# 7. Save hasil ke data/processed
# =========================
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)

print("NLP Scoring selesai!")
print("File tersimpan di data/processed/")
