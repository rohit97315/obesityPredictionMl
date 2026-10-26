import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_model(data_path):
    # Load dataset via cleaning module
    from .data_loader import load_and_clean_data
    df, label_encoders = load_and_clean_data(data_path)

    # Separate Features and Target
    X = df.drop(columns=['NObeyesdad', 'NObeyesdad_encoded'])
    y = df['NObeyesdad_encoded']

    # Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    target_names = label_encoders['target'].classes_

    report = classification_report(y_test, y_pred, target_names=target_names)
    
    print("="*60)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("="*60)
    print("Classification Report:\n", report)

    # Save Model Artifacts
    joblib.dump(model, 'models/obesity_rf_model.joblib')
    
    with open('outputs/model_evaluation.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n\n" + report)

    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_model('data/ObesityDataSet_raw_and_data_sinthetic.csv')