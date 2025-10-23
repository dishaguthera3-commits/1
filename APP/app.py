import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(
    page_title="AI BMI Health & Diet Recommender",
    page_icon="💪",
    layout="wide"
)

st.title("🌱 AI BMI Health & Diet Recommender (SDG 3)")
st.markdown("This app calculates your **BMI** and recommends **diet and activity plans** using ML models trained on real data.")

# --------------------------
# Load ML Models
# --------------------------
diet_model = joblib.load("diet_model.pkl")
activity_model = joblib.load("activity_model.pkl")

le_activity = joblib.load("le_activity.pkl")
le_diet = joblib.load("le_diet.pkl")
le_rec_diet = joblib.load("le_rec_diet.pkl")
le_rec_activity = joblib.load("le_rec_activity.pkl")

# --------------------------
# Sidebar: User Inputs
# --------------------------
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age", 10, 100, 25)
height = st.sidebar.number_input("Height (cm)", 100, 220, 170)
weight = st.sidebar.number_input("Weight (kg)", 30, 150, 65)
activity = st.sidebar.selectbox("Activity Level", ["Low", "Moderate", "High"])
diet = st.sidebar.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"])
sleep = st.sidebar.slider("Average Sleep Hours", 3, 12, 7)

# --------------------------
# BMI Calculation
# --------------------------
bmi = round(weight / ((height/100)**2), 2)

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "blue"
    elif bmi < 25: return "Normal weight", "green"
    elif bmi < 30: return "Overweight", "orange"
    else: return "Obese", "red"

category, color = bmi_category(bmi)
st.markdown(f"<h2 style='color:{color}'>Your BMI: {bmi} ({category})</h2>", unsafe_allow_html=True)

# --------------------------
# Prepare input for ML
# --------------------------
activity_enc = le_activity.transform([activity])[0]
diet_enc = le_diet.transform([diet])[0]

input_df = pd.DataFrame([[age, bmi, activity_enc, sleep, diet_enc]],
                        columns=['Age','BMI','ActivityLevel_enc','SleepHours','DietType_enc'])

# --------------------------
# ML Predictions
# --------------------------
pred_diet_enc = diet_model.predict(input_df)[0]
pred_diet = le_rec_diet.inverse_transform([pred_diet_enc])[0]

pred_activity_enc = activity_model.predict(input_df)[0]
pred_activity = le_rec_activity.inverse_transform([pred_activity_enc])[0]

# --------------------------
# Display Recommendations
# --------------------------
st.header("💡 Personalized Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Diet Plan")
    st.info(pred_diet)

with col2:
    st.subheader("Activity Plan")
    st.success(pred_activity)

# --------------------------
# Optional: Show Training Data Insights
# --------------------------
if st.checkbox("Show Training Data Overview"):
    st.subheader("Sample Training Data")
    data = pd.read_csv("health_data_recommender.csv")
    st.dataframe(data.head(10))

    st.subheader("BMI Distribution in Training Data")
    st.bar_chart(data['BMI'])

st.markdown("---")
st.caption("Developed as a school AI project for SDG 3: Good Health & Well-Being")
