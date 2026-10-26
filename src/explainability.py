import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def generate_shap_explanations(model, X_train, X_test, target_names):
    """
    Computes SHAP values for global and local model interpretability.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Global Interpretability: Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, class_names=target_names, show=False)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=300)
    plt.close()
    print("[+] Saved SHAP Global Feature Importance plot to 'outputs/feature_importance.png'")

    return explainer, shap_values

def explain_individual_prediction(explainer, model, patient_df, feature_names, target_names):
    """
    Generates single-patient dynamic decision interpretation.
    """
    pred_idx = model.predict(patient_df)[0]
    pred_class = target_names[pred_idx]
    
    patient_shap = explainer.shap_values(patient_df)
    
    # Handle both list output (multi-class) and array output from shap
    if isinstance(patient_shap, list):
        class_shap = patient_shap[pred_idx][0]
    elif len(patient_shap.shape) == 3:
        # Shape: (samples, features, classes) or (classes, samples, features)
        # For single sample, usually (classes, 1, features) or (1, features, classes)
        class_shap = patient_shap[pred_idx][0] if patient_shap.shape[0] == len(target_names) else patient_shap[0, :, pred_idx]
    else:
        class_shap = patient_shap[0]

    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': patient_df.iloc[0].values,
        'SHAP_Contribution': class_shap
    }).sort_values(by='SHAP_Contribution', key=abs, ascending=False)

    return pred_class, contrib_df