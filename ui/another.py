import streamlit as st
import pickle
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

# 🎯 Load trained model & encoders
MODEL_PATH = r"C:\Users\HP\Documents\GitHub\Coding-week-test\notebooks\model.pkl"
SHAP_PATH = r"C:\Users\HP\Documents\GitHub\Coding-week-test\notebooks\shap_explainer.pkl"
ENCODER_PATH = r"C:\Users\HP\Documents\GitHub\Coding-week-test\notebooks\label_encoder.pkl"
BACKGROUND_PATH = r"C:\Users\HP\Desktop\Adobe.png"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SHAP_PATH, "rb") as file:
    explainer = pickle.load(file)

with open(ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

import base64

# ✅ Function to add background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
add_bg_from_local(BACKGROUND_PATH)

# ✅ Apply CSS for Blurred Background for UI Sections
st.markdown(
    """
    <style>
    .blurred-box {
        background: rgba(255, 255, 255, 0.7); /* Semi-transparent background */
        backdrop-filter: blur(10px); /* Blur effect */
        padding: 20px; /* Add some padding */
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Soft shadow */
        margin-bottom: 20px; /* Add spacing */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ **Header (No Duplicate)**
st.markdown("<h1 style='text-align: center;'>🩺 Medical Decision Support 🩺  Obesity Risk Estimator</h1>", unsafe_allow_html=True)



# ✅ Patient Data Section
st.markdown('<div class="blurred-box">', unsafe_allow_html=True)
st.markdown("<h3>📝 Enter Patient Information</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("📅 Age", min_value=10, max_value=100, value=30, key="age_input")
    weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70, key="weight_input")
    height = st.number_input("📏 Height (cm)", min_value=100, max_value=220, value=170, key="height_input")

# ✅ Ensure `key` values are unique to avoid conflicts

with col2:
    fcvc = st.slider("🥦 Vegetable Consumption (FCVC)", min_value=0.0, max_value=5.0, value=2.5, step=0.01)
    ncp = st.slider("🍽️ Main Meals per Day (NCP)", min_value=1, max_value=6, value=3, step=1)
    ch2o = st.slider("💧 Daily Water Intake (CH2O)", min_value=0.0, max_value=5.0, value=2.5, step=0.01)
    faf = st.slider("🏃 Physical Activity (FAF)", min_value=0.0, max_value=7.0, value=3.0, step=0.01)
    tue = st.slider("📱 Time Using Technology (TUE)", min_value=0.0, max_value=10.0, value=2.0, step=0.01)

# ✅ Create Three Columns (Left, Middle for Lifestyle & Habits, Right)
col1, col2, col3 = st.columns([4, 6, 4])  # Middle column is wider

# 🎯 Middle Column: Lifestyle & Habits
with col2:
    st.markdown("<h3 style='text-align: center;'>🏷️ Lifestyle & Habits</h3>", unsafe_allow_html=True)

    gender = st.radio("👤 Gender", ["Male", "Female"], horizontal=True)
    family_history = st.radio("👨‍👩‍👧 Family History of Obesity?", ["Yes", "No"], horizontal=True)
    favc = st.radio("🍔 High-Calorie Food Consumption (FAVC)?", ["Yes", "No"], horizontal=True)
    caec = st.selectbox("🍪 Eating Between Meals (CAEC)?", ["No", "Sometimes", "Frequently", "Always"])
    smoking = st.radio("🚬 Smoker?", ["Yes", "No"], horizontal=True)
    scc = st.radio("📊 Calories Tracking (SCC)?", ["Yes", "No"], horizontal=True)
    calc = st.selectbox("🍷 Alcohol Consumption (CALC)?", ["No", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("🛵 Main Transportation Mode", ["Walking", "Bike", "Motorbike", "Public Transport", "Automobile"])
    
# ✅ Convert categorical inputs
gender_map = {"Male": 0, "Female": 1}
binary_map = {"Yes": 1, "No": 0}
favc_map = {"Yes": 1, "No": 0}
caec_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
scc_map = {"Yes": 1, "No": 0}
calc_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {"Walking": 0, "Bike": 1, "Motorbike": 2, "Public Transport": 3, "Automobile": 4}

# ✅ Feature Matching
input_data = np.array([[gender_map[gender], age, height, weight, 
                        binary_map[family_history], favc_map[favc], fcvc, ncp, 
                        caec_map[caec], binary_map[smoking], ch2o, scc_map[scc], 
                        faf, tue, calc_map[calc], mtrans_map[mtrans]]])

# ✅ Prediction Button
if st.button("🔍 Predict Obesity Level & View Results"):
    st.session_state["prediction_triggered"] = True  # ✅ Save state to show results

coded = {
    'Insufficient Weight': 0, 'Normal Weight': 1, 'Overweight Level I': 2, 
    'Overweight Level II': 3, 'Obesity Type I': 4, 'Obesity Type II': 5, 'Obesity Type III': 6
}

# ✅ Show Results Below the Button
if st.session_state.get("prediction_triggered", False):
    st.markdown("---")
    st.markdown("<h3>📊 Prediction Results</h3>", unsafe_allow_html=True)

    # 🎯 Prediction Display
    prediction = model.predict(input_data)[0]
    predicted_label = next(k for k, v in coded.items() if v == prediction)

    st.markdown(f"<h3 style='color:green;'>🩺 Prediction: {predicted_label}</h3>", unsafe_allow_html=True)

    
    # SHAP Explanation
    st.subheader("🧐 Why This Prediction?")

    feature_names = [
            "Gender", "Age", "Height", "Weight", "Family History", "FAVC",
            "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS" ]  # ✅ Now all features are included

    shap_values = explainer.shap_values(input_data)
    # ✅ Apply Full-Width CSS
    # 🎯 SHAP Explanation Section
    st.subheader("📊 SHAP Explanation - Feature Importance")

    # ✅ SHAP Graph (Centered, Full Width)
    shap.initjs()
    plt.figure(figsize=(12, 6))  # Increased figure size for clarity
    shap.summary_plot(shap_values, pd.DataFrame(input_data, columns=feature_names), show=False)
    st.pyplot(plt)

# ✅ Explanation Below the Graph (Restored)
    with st.expander("🔍 Understanding the Graph", expanded=False):
        st.markdown("""
### 🔍 Key Components of the Graph:
1️⃣ **X-Axis (SHAP Value)**
- **Negative values (left)**: The feature decreases the obesity level prediction.
- **Positive values (right)**: The feature increases the obesity level prediction.

2️⃣ **Y-Axis (Feature Name)**
- This graph shows the most influential features in this prediction.

3️⃣ **Dots (Data Points)**
- **Dots above 0** → The feature **increased** the obesity prediction.
- **Dots below 0** → The feature **decreased** the obesity prediction.
""", unsafe_allow_html=True)

# ✅ Feature Legend Below the Explanation (Well-Formatted)
    with st.expander("📌 Feature Legend", expanded=False):
        legend_dict = {
    "FAVC": "Do you eat high-calorie food frequently?",
    "FCVC": "Do you usually eat vegetables in your meals?",
    "NCP": "How many main meals do you have daily?",
    "CAEC": "Do you eat any food between meals?",
    "SMOKE": "Do you smoke?",
    "CH2O": "How much water do you drink daily?",
    "SCC": "Do you monitor the calories you eat daily?",
    "FAF": "How often do you have physical activity?",
    "TUE": "How much time do you use technological devices?",
    "CALC": "How often do you drink alcohol?",
    "MTRANS": "Which transportation do you usually use?",
}

# ✅ Display the legend properly formatted below the explanation
        for key, value in legend_dict.items():
            st.markdown(f"- **{key}**: {value}")

    
    # ✅ Feedback Section for Doctor
    st.markdown("---")
    st.markdown("<h3>📝 Doctor's Feedback</h3>", unsafe_allow_html=True)
    
    feedback = st.radio("Was the prediction accurate?", ["✅ Yes, it was correct", "❌ No, it was incorrect"], horizontal=True)
    
    correction = ""
    if feedback == "❌ No, it was incorrect":
        correction = st.text_area("Please provide the correct obesity level or any feedback:")

    if st.button("Submit Feedback"):
        try:
            with open("doctor_feedback.txt", "a", encoding="utf-8") as f:
                f.write(f"Prediction: {predicted_label} | Doctor Feedback: {feedback} | Correction: {correction if feedback == '❌ No, it was incorrect' else 'N/A'}\n")
            st.success("✅ Thank you! Your feedback has been recorded.")
        except Exception as e:
            st.error(f"⚠️ Error saving feedback: {e}")
