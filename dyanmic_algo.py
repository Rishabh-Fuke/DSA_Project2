import pandas as pd
import numpy as np

def dp_allocation(df, icu_capacity=10000, doctor_capacity=40000):
    """
    Dynamic Programming approach for optimal patient allocation
    given limited resource time capacities.
    """

    # Separate ICU and Doctor patients
    icu_patients = df[df["resource_type"] == "ICU"].reset_index(drop=True)
    doc_patients = df[df["resource_type"] == "Doctor"].reset_index(drop=True)

    def knapsack(patients, capacity):
        n = len(patients)
        durations = patients["treatment_duration"].tolist()
        urgencies = patients["urgency_score"].tolist()

        # DP table: dp[i][w] = max urgency achievable using first i patients and total time w
        dp = np.zeros((n + 1, capacity + 1))

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if durations[i - 1] <= w:
                    dp[i][w] = max(
                        urgencies[i - 1] + dp[i - 1][w - durations[i - 1]],
                        dp[i - 1][w]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        # Backtrack to find selected patients
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.append(patients.loc[i - 1, "patient_id"])
                w -= durations[i - 1]

        return selected, dp[n][capacity]

    # Run DP for each resource
    icu_selected, icu_score = knapsack(icu_patients, icu_capacity)
    doc_selected, doc_score = knapsack(doc_patients, doctor_capacity)

    # Combine results
    assigned = set(icu_selected + doc_selected)
    df["status"] = df["patient_id"].apply(lambda x: "Assigned" if x in assigned else "Waiting")

    total_waiting_time = df[df["status"] == "Waiting"]["treatment_duration"].sum()
    utilization_rate = len(assigned) / len(df)

    return {
        "assignments": df,
        "icu_score": icu_score,
        "doc_score": doc_score,
        "waiting_time": total_waiting_time,
        "utilization": utilization_rate
    }

# TESTING ON patient_data_short.csv

if __name__ == "__main__":

    try:
        # 1️⃣ Load your dataset
        df = pd.read_csv("patient_data_short.csv")
        print("✅ Loaded dataset successfully!")
        print(df.head(), "\n")

        # 2️⃣ Run Dynamic Programming allocation
        print("🏥 Running DP allocation...\n")
        results = dp_allocation(df, icu_capacity=1000, doctor_capacity=2000)

        # 3️⃣ Display key results
        print("=== RESULTS ===")
        print(f"ICU Score (total urgency served): {results['icu_score']:.2f}")
        print(f"Doctor Score (total urgency served): {results['doc_score']:.2f}")
        print(f"Total Waiting Time: {results['waiting_time']:.2f}")
        print(f"Utilization Rate: {results['utilization']*100:.2f}%")

        # 4️⃣ Preview assignments
        print("\nSample Allocation Table:")
        print(results["assignments"].head(10))

    except FileNotFoundError:
        print("❌ Could not find 'patient_data_short.csv'. Make sure it's in the same folder as this file.")
    except Exception as e:
        print("⚠️ An error occurred while testing:")
        print(e)
