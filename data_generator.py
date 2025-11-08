import pandas as pd
import numpy as np

def generate_patient_data(n=300, seed=42):
    np.random.seed(seed)
    patient_ids = np.arange(1, n + 1)
    urgency_scores = np.round(np.random.uniform(1, 10, n), 1)
    arrival_times = pd.date_range("2025-01-01", periods=n, freq="min")
    treatment_durations = np.random.randint(30, 240, n)  # in minutes
    resource_types = np.random.choice(["ICU", "Doctor"], size=n, p=[0.3, 0.7])

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "urgency_score": urgency_scores,
        "arrival_time": arrival_times,
        "treatment_duration": treatment_durations,
        "resource_type": resource_types
    })

    df.to_csv("patient_data_short.csv", index=False)
    print(f"✅ Generated {n} rows and saved to patient_data.csv")

if __name__ == "__main__":
    generate_patient_data()