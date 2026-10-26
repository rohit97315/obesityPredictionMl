import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_clean_data(file_path):
    """
    Loads raw CSV dataset, performs cleaning, feature engineering, and encoding.
    """
    df = pd.read_csv(file_path)

    # 1. Clean duplicates & handle missing values
    df = df.drop_duplicates()
    df = df.dropna()

    # 2. Feature Engineering
    # Calculate BMI: Weight (kg) / Height (m)^2
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # Convert continuous continuous-like features (due to synthetic generation) to rounded metrics where relevant
    df['FCVC_round'] = df['FCVC'].round()  # Vegetable consumption frequency
    df['NCP_round'] = df['NCP'].round()    # Main meals count
    df['CH2O_round'] = df['CH2O'].round()  # Water intake
    df['FAF_round'] = df['FAF'].round()    # Physical activity frequency
    df['TUE_round'] = df['TUE'].round()    # Screen time / tech usage

    # 3. Categorical Encodings
    cat_columns = [
        'Gender', 'family_history_with_overweight', 'FAVC', 
        'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]

    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 4. Target Variable Encoding
    target_le = LabelEncoder()
    df['NObeyesdad_encoded'] = target_le.fit_transform(df['NObeyesdad'])
    label_encoders['target'] = target_le

    # Save encoders for downstream deployment
    joblib.dump(label_encoders, 'models/label_encoders.joblib')

    return df, label_encoders

if __name__ == "__main__":
    df, encoders = load_and_clean_data('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    print(f"Data cleaned successfully. Shape: {df.shape}")