import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from src.risk_engine import ObesityRiskAssessmentEngine

# Page Config
st.set_page_config(
    page_title="Obesity Risk Assessment & Early Intervention Portal",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/obesity_rf_model.joblib')
    encoders = joblib.load('models/label_encoders.joblib')
    return model, encoders

model, encoders = load_artifacts()
target_names = encoders['target'].classes_
explainer = shap.TreeExplainer(model)

st.title("🩺 Interpretable & Risk-Aware Obesity Assessment Framework")
st.markdown("Enter patient metrics, lifestyle factors, and eating habits below to receive an instant risk prediction and personalized health recommendations.")

with st.form("patient_form"):
    st.subheader("Patient Clinical & Lifestyle Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        age = st.number_input("Age", min_value=10.0, max_value=100.0, value=25.0)
        height = st.number_input("Height (meters)", min_value=1.0, max_value=2.5, value=1.75)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0)
        family_history = st.selectbox("Family History with Overweight", encoders['family_history_with_overweight'].classes_)
        favc = st.selectbox("Frequent High-Calorie Food Consumption (FAVC)", encoders['FAVC'].classes_)

    with col2:
        fcvc = st.slider("Vegetable Consumption Frequency (FCVC: 1=Low, 3=High)", 1.0, 3.0, 2.0)
        ncp = st.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0)
        caec = st.selectbox("Eating Between Meals (CAEC)", encoders['CAEC'].classes_)
        smoke = st.selectbox("Smoking Status (SMOKE)", encoders['SMOKE'].classes_)
        ch2o = st.slider("Daily Water Consumption (CH2O: Liters)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Calories Consumption Tracking (SCC)", encoders['SCC'].classes_)

    with col3:
        faf = st.slider("Physical Activity Frequency (FAF: 0=None, 3=High)", 0.0, 3.0, 1.0)
        tue = st.slider("Tech/Screen Time Usage (TUE: Hours)", 0.0, 2.0, 1.0)
        calc = st.selectbox("Alcohol Consumption (CALC)", encoders['CALC'].classes_)
        mtrans = st.selectbox("Transportation Used (MTRANS)", encoders['MTRANS'].classes_)

    submitted = st.form_submit_button("Generate Risk Assessment")

if submitted:
    # 1. Process inputs into DataFrame matching training columns
    bmi = weight / (height ** 2)
    
    input_data = pd.DataFrame({
        'Gender': [encoders['Gender'].transform([gender])[0]],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [encoders['family_history_with_overweight'].transform([family_history])[0]],
        'FAVC': [encoders['FAVC'].transform([favc])[0]],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [encoders['CAEC'].transform([caec])[0]],
        'SMOKE': [encoders['SMOKE'].transform([smoke])[0]],
        'CH2O': [ch2o],
        'SCC': [encoders['SCC'].transform([scc])[0]],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [encoders['CALC'].transform([calc])[0]],
        'MTRANS': [encoders['MTRANS'].transform([mtrans])[0]],
        'BMI': [bmi],
        'FCVC_round': [round(fcvc)],
        'NCP_round': [round(ncp)],
        'CH2O_round': [round(ch2o)],
        'FAF_round': [round(faf)],
        'TUE_round': [round(tue)]
    })

    # 2. Predict & Explain
    pred_idx = model.predict(input_data)[0]
    pred_class = target_names[pred_idx]
    
    patient_shap = explainer.shap_values(input_data)
    
    # Robust SHAP handling across various library versions
    if isinstance(patient_shap, list):
        class_shap = patient_shap[pred_idx][0]
    elif len(patient_shap.shape) == 3:
        if patient_shap.shape[2] == len(target_names):
            class_shap = patient_shap[0, :, pred_idx]
        else:
            class_shap = patient_shap[0, pred_idx, :]
    else:
        class_shap = patient_shap[0]
    
    contrib_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Value': input_data.iloc[0].values,
        'SHAP_Contribution': class_shap
    }).sort_values(by='SHAP_Contribution', key=abs, ascending=False)

    # 3. Generate Risk Assessment
    risk_engine = ObesityRiskAssessmentEngine()
    report = risk_engine.generate_assessment(input_data, pred_class, contrib_df)

    # 4. Display UI Results
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("📊 Diagnostic Prediction")
        st.metric(label="Calculated BMI", value=f"{bmi:.2f}")
        st.metric(label="Predicted Obesity Category", value=report['predicted_category'])
        st.error(f"**Risk Level:** {report['risk_level']} (Tier {report['risk_score_tier']})")

    with res_col2:
        st.subheader("💡 Actionable Recommendations")
        for rec in report['personalized_interventions']:
            st.success(rec)

    st.markdown("---")
    st.subheader("🔍 Top Contributing Factors (Interpretable AI Insights)")
    st.dataframe(contrib_df.head(5)[['Feature', 'Value', 'SHAP_Contribution']], use_container_width=True)