import numpy as np
from src.fis.fis_model import FISChatbot

# Load data (hasil Step 2)
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

fis = FISChatbot()

# Contoh prediksi satu sampel
idx = 0
x_sample = X_train[idx]
y_true = y_train[idx]
y_pred = fis.predict_one(x_sample)
y_pred_label = fis.predict_one(x_sample, return_label=True)

print("Input scores     :", x_sample)
print("True tone index  :", y_true)
print("Pred tone index  :", y_pred)
print("Pred tone label  :", y_pred_label)
