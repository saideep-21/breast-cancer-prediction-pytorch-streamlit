import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# load model and scaler 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 30
hidden_size = 64
output_size = 1

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("path of the breast_cancer_model.pth", map_location=device)) #paste the path of file here
model.eval()

scaler = joblib.load("path of the scaler.pkl") #paste the path of file here


# Prediction function

def predict_single(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        prob = model(tensor).item()
    return "Benign" if prob >= 0.5 else "Malignant"

def predict_batch(df):
    features = scaler.transform(df)
    tensor = torch.tensor(features, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(tensor).cpu().numpy().flatten()
    preds = ["Benign" if p >= 0.5 else "Malignant" for p in probs]
    return preds


# Streamlit UI

st.title("Breast Cancer Prediction App")
st.write("Choose between **Single Prediction** (manual input) or **Batch Prediction** (CSV upload).")

mode = st.radio("Select Mode:", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    st.subheader("🔹 Single Prediction")
    st.write("Enter 30 comma-separated feature values below (same order as dataset).")

    user_input = st.text_area("Enter features:", "")
    if st.button("Predict"):
        try:
            features = [float(x.strip()) for x in user_input.split(",")]
            if len(features) != 30:
                st.error("❌ Please enter exactly 30 values!")
            else:
                result = predict_single(features)
                st.success(f"Prediction: **{result}**")
        except:
            st.error("⚠️ Invalid input. Make sure all 30 values are numbers.")

elif mode == "Batch Prediction":
    st.subheader("🔹 Batch Prediction")
    st.write("Upload a CSV file where **each row contains 30 features**.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] != 30:
                st.error("❌ CSV must have exactly 30 columns.")
            else:
                preds = predict_batch(df.values)
                df["Prediction"] = preds
                st.write("✅ Predictions:")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
