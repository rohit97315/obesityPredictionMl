import os
import joblib
import pandas as pd
from src.data_loader import load_and_clean_data
from src.train import train_model
from src.explainability import generate_shap_explanations, explain_individual_prediction
from src.risk_engine import ObesityRiskAssessmentEngine

def main():
    # Setup directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    dataset_path = 'data/ObesityDataSet_raw_and_data_sinthetic.csv'

    print("\n--- 1. Training & Evaluating Obesity Prediction Model ---")
    model, X_train, X_test, y_train, y_test = train_model(dataset_path)

    encoders = joblib.load('models/label_encoders.joblib')
    target_names = encoders['target'].classes_

    print("\n--- 2. Computing SHAP Model Interpretability ---")
    explainer, shap_values = generate_shap_explanations(model, X_train, X_test, target_names)

    print("\n--- 3. Running Actionable Risk Assessment on Sample Patient ---")
    sample_patient = X_test.iloc[[0]]  # Pick first sample from test set
    
    pred_class, top_contribs = explain_individual_prediction(
        explainer, model, sample_patient, X_test.columns, target_names
    )

    risk_engine = ObesityRiskAssessmentEngine()
    report = risk_engine.generate_assessment(sample_patient, pred_class, top_contribs)

    print("\n========================================================")
    print("           PERSONALIZED HEALTH RISK ASSESSMENT           ")
    print("========================================================")
    print(f"Predicted Obesity Class : {report['predicted_category']}")
    print(f"Risk Assessment Level  : {report['risk_level']} ({report['risk_score_tier']})")
    print("\nTop Contributing Risk Drivers:")
    for driver in report['key_risk_drivers']:
        print(f"  • {driver['Feature']} = {driver['Value']:.2f} (Impact: {driver['SHAP_Contribution']:+.4f})")
        
    print("\nRecommended Personalized Interventions:")
    for intervention in report['personalized_interventions']:
        print(f"  {intervention}")
    print("========================================================\n")

if __name__ == "__main__":
    main()