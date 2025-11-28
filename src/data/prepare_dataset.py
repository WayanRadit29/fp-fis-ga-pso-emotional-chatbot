# src/data/prepare_dataset.py

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset dari Hugging Face (dair-ai/emotion)
print("Loading dataset...")
dataset = load_dataset("dair-ai/emotion")

# 2. Label → nama emosi
emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 3. Mapping emosi → tone chatbot
tone_map = {
    "sadness": "empathetic",
    "anger": "calming",
    "fear": "supportive",
    "joy": "friendly",
    "love": "warm",
    "surprise": "informative"
}

def map_label_to_tone(label_id):
    emotion = emotion_map[label_id]
    return tone_map[emotion]

# 4. Ambil semua data train dari dataset HF
texts = dataset["train"]["text"]
labels = dataset["train"]["label"]

# 5. Konversi label → tone
tones = [map_label_to_tone(lbl) for lbl in labels]

# 6. Split train/test (80/20)
train_texts, test_texts, train_tones, test_tones = train_test_split(
    texts,
    tones,
    test_size=0.2,
    stratify=tones,
    random_state=42
)

# 7. Convert jadi DataFrame
df_train = pd.DataFrame({"text": train_texts, "tone": train_tones})
df_test = pd.DataFrame({"text": test_texts, "tone": test_tones})

# 8. Save ke folder processed
df_train.to_csv("../../data/processed/train_tone.csv", index=False)
df_test.to_csv("../../data/processed/test_tone.csv", index=False)


print("Dataset selesai diproses!")
print("File tersimpan di data/processed/")
