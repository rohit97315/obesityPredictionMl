import pandas as pd

class ObesityRiskAssessmentEngine:
    def __init__(self):
        # Risk Mapping Dictionary
        self.category_risk_map = {
            'Insufficient_Weight': ('Low Risk', 1),
            'Normal_Weight': ('Optimal Health Risk', 0),
            'Overweight_Level_I': ('Moderate Risk', 2),
            'Overweight_Level_II': ('Elevated Risk', 3),
            'Obesity_Type_I': ('High Risk', 4),
            'Obesity_Type_II': ('Very High Risk', 5),
            'Obesity_Type_III': ('Critical Risk', 6)
        }

    def generate_assessment(self, patient_features: pd.DataFrame, predicted_class: str, feature_contributions: pd.DataFrame):
        row = patient_features.iloc[0]
        risk_label, risk_score = self.category_risk_map.get(predicted_class, ('Unknown', 0))

        recommendations = []

        # Rule-based / Feature Contribution Driven Interventions
        if row['BMI'] >= 25.0:
            recommendations.append("🎯 Weight Management: Target a gradual weight reduction (0.5-1.0 kg/week) under medical guidance.")

        if row['FAF'] < 1.0: # Physical Activity Frequency
            recommendations.append("🏋️ Physical Activity: Increase moderate aerobic activity (e.g., brisk walking) to at least 150 mins/week.")

        if row['CH2O'] < 2.0: # Water Intake
            recommendations.append("💧 Hydration: Elevate daily water consumption to at least 2.0 - 2.5 Liters per day.")

        if row['FCVC'] < 2.0: # Vegetable Intake
            recommendations.append("🥗 Nutrition: Increase vegetable consumption to 3+ servings daily.")

        if row['TUE'] > 2.0: # Screen Time
            recommendations.append("📱 Screen Time: Reduce non-work screen exposure to under 2 hours daily.")

        if row['FAVC'] == 1: # High Caloric Food
            recommendations.append("🍔 Dietary Quality: Limit frequent intake of high-caloric processed foods.")

        if row['SMOKE'] == 1:
            recommendations.append("🚭 Lifestyle: Implement a smoking cessation protocol.")

        assessment_report = {
            "predicted_category": predicted_class,
            "risk_level": risk_label,
            "risk_score_tier": f"{risk_score}/6",
            "key_risk_drivers": feature_contributions.head(4).to_dict(orient='records'),
            "personalized_interventions": recommendations
        }

        return assessment_report