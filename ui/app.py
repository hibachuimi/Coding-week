import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
MODEL_PATH = r"C:\Users\HP\Documents\GitHub\Coding-week-test\notebooks\model.pkl"
SHAP_PATH = r"C:\Users\HP\Documents\GitHub\Coding-week-test\notebooks\shap_explainer.pkl"
ENCODER_PATH = r"C:\Users\HP\Documents\GitHub\Coding-week-test\notebooks\label_encoder.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SHAP_PATH, "rb") as file:
    explainer = pickle.load(file)

with open(ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)  # Load encoder to decode obesity levels

# UI Header
st.title("🔬 Medical Decision Support - Obesity Risk Estimator")

st.markdown("""
This application helps physicians estimate the **obesity risk** of patients based on their **lifestyle and physical condition**.  
The predictions are backed by **explainable AI** using **SHAP analysis**.
""")

# Collect Patient Data (Only Using the Features Model Was Trained On)
st.header("📝 Enter Patient Information")

# ✅ Numeric Inputs (Training Features Only)
age = st.number_input("Age", min_value=10, max_value=100)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
height = st.number_input("Height (cm)", min_value=100, max_value=220)
fcvc = st.slider("Frequency of Vegetable Consumption (FCVC)", min_value=1.0, max_value=3.0, step=0.1)
ncp = st.slider("Number of Main Meals (NCP)", min_value=1, max_value=4, step=1)
ch2o = st.slider("Daily Water Intake (CH2O)", min_value=1.0, max_value=3.0, step=0.1)
faf = st.slider("Physical Activity Frequency (FAF)", min_value=0.0, max_value=3.0, step=0.1)
tue = st.slider("Time Using Technology (TUE)", min_value=0.0, max_value=2.0, step=0.1)

# ✅ Categorical Inputs (Training Features Only)
gender = st.radio("Gender", ["Male", "Female"])
family_history = st.radio("Family History of Obesity?", ["Yes", "No"])
favc = st.radio("Frequent High-Calorie Food Consumption (FAVC)?", ["Yes", "No"])
caec = st.selectbox("Eating Between Meals (CAEC)?", ["No", "Sometimes", "Frequently", "Always"])
smoking = st.radio("Smoker?", ["Yes", "No"])
scc = st.radio("Calories Tracking (SCC)?", ["Yes", "No"])
calc = st.selectbox("Alcohol Consumption (CALC)?", ["No", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Main Transportation Mode", ["Walking", "Bike", "Motorbike", "Public Transport", "Automobile"])

# Convert categorical inputs to match training encoding
gender_map = {"Male": 0, "Female": 1}
binary_map = {"Yes": 1, "No": 0}
favc_map = {"Yes": 1, "No": 0}
caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
scc_map = {"Yes": 1, "No": 0}
calc_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {"Walking": 0, "Bike": 1, "Motorbike": 2, "Public Transport": 3, "Automobile": 4}

# ✅ Fixed Input Data - Matches Training Exactly (16 Features)
input_data = np.array([[gender_map[gender], age, height, weight, 
                        binary_map[family_history], favc_map[favc], fcvc, ncp, 
                        caec_map[caec], binary_map[smoking], ch2o, scc_map[scc], 
                        faf, tue, calc_map[calc], mtrans_map[mtrans]]])  # ✅ 16 features exactly

# Debugging - Print feature count
st.write(f"✅ Input Data Shape: {input_data.shape} (Expected: {model.n_features_in_})")

# ✅ Mapping of Encoded Values to Labels
coded = {
    'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2, 
    'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6
}

# Prediction Button
if st.button("🔍 Predict Obesity Level"):
    try:
        prediction = model.predict(input_data)[0]  # Get single prediction
        predicted_label = next(k for k, v in coded.items() if v == prediction)  # ✅ Decode prediction

        # Display the actual obesity level name
        st.success(f"🩺 **Predicted Obesity Level:** {predicted_label}")

        # SHAP Explanation
        st.subheader("🧐 Why This Prediction?")

        feature_names = [
            "Gender", "Age", "Height", "Weight", "Family History", "FAVC",
            "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
        ]  # ✅ Now all features are included

        shap_values = explainer.shap_values(input_data)

        # SHAP Plot with Correct Feature Names
        shap.initjs()
        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, pd.DataFrame(input_data, columns=feature_names), show=False)
        st.pyplot(plt)

        st.write("🔹 This graph shows how each feature contributed to the prediction.")

    except Exception as e:
        st.error(f"⚠️ Error making prediction: {e}")

st.markdown("---")
st.markdown("📌 *Built for Medical Decision Support - Coding Week 2025*")

# ✅ FIXED: Convert height to meters before calculating BMI
bmi = weight / ((height / 100) ** 2)

st.write(f"📌 Calculated BMI : *{bmi:.2f}*")

if bmi < 18.5:
    st.write("🔹 Catégorie IMC : *Poids Insuffisant*")
elif 18.5 <= bmi < 25:
    st.write("🔹 Catégorie IMC : *Poids Normal*")
elif 25 <= bmi < 30:
    st.write("🔹 Catégorie IMC : *Surpoids*")
elif 30 <= bmi < 35:
    st.write("🔹 Catégorie IMC : *Obésité Type I*")
elif 35 <= bmi < 40:
    st.write("🔹 Catégorie IMC : *Obésité Type II*")
else:
    st.write("🔹 Catégorie IMC : *Obésité Type III*")
