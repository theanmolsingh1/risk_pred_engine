import numpy as np
import pandas as pd
import joblib  # use joblib, not pickle
# -----------------------------
# 1️⃣ Load the model saved with joblib
# -----------------------------
model = joblib.load("final_lgb_model.pkl")  # replace with your joblib file name
print(f"Model loaded successfully! Number of features expected: {model.n_features_in_}")
# -----------------------------
# 2️⃣ Input for 1 patient
# -----------------------------
patient_input = np.array([1,1,20,1,1,
                          1,1,1,1,1,0.56,0.11,1,0.99,0.12])
# Pad with random values if model expects more features
while patient_input.shape[0] < model.n_features_in_:
    patient_input = np.append(patient_input, np.random.uniform(0,1))
X = patient_input.reshape(1, -1)
# -----------------------------
# 3️⃣ Make prediction
# -----------------------------
pred_class = model.predict(X)[0]
pred_prob = model.predict_proba(X)[0]
# -----------------------------
# 4️⃣ Show only predicted class and probability
# -----------------------------
output = pd.DataFrame({
    "predicted_class": [pred_class],
    "prob_class_0": [pred_prob[0]],
    "prob_class_1": [pred_prob[1]]
})
print(output)
