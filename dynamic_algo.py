import pandas as pd
import numpy as np
from datetime import timedelta
import time
import warnings
warnings.filterwarnings("ignore")


class TimeIndexedDP:
    """
    Time-indexed Dynamic Programming for patient scheduling.
    Considers both urgency and wait time in allocation.
    """

    def __init__(self, num_doctors=10, num_icu=5, total_time_hours=8, slot_minutes=15, alpha=0.2):
        self.num_doctors = num_doctors
        self.num_icu = num_icu
        self.total_time_hours = total_time_hours
        self.slot_minutes = slot_minutes
        self.num_slots = (total_time_hours * 60) // slot_minutes
        self.alpha = alpha # weight for wait time in priority score

    def allocate_resources(self, df):
        """Main allocation logic using time-indexed DP."""
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        df = df.sort_values('arrival_time').reset_index(drop=True)

        sim_start = df['arrival_time'].min()
        sim_end = sim_start + timedelta(hours=self.total_time_hours)
        df = df[df['arrival_time'] <= sim_end].copy()

        # Convert to time slots
        df['arrival_slot'] = df['arrival_time'].apply(
            lambda x: int((x - sim_start).total_seconds() / 60 / self.slot_minutes)
        )
        df['duration_slots'] = df['treatment_duration'].apply(
            lambda x: max(1, int(np.ceil(x / self.slot_minutes)))
        )

        # Separate by resource type
        doctor_patients = df[df['resource_type'] == 'Doctor'].reset_index(drop=True)
        icu_patients = df[df['resource_type'] == 'ICU'].reset_index(drop=True)

        # Run DP for each resource
        doc_assignments, doc_urgency = self._run_dp_for_resource(doctor_patients, self.num_doctors)
        icu_assignments, icu_urgency = self._run_dp_for_resource(icu_patients, self.num_icu)

        all_assignments = doc_assignments + icu_assignments
        total_urgency = doc_urgency + icu_urgency

        # Calculate metrics
        metrics = self._calculate_metrics(df, all_assignments, sim_start, sim_end, total_urgency)
        return metrics, all_assignments

    def _run_dp_for_resource(self, patients, num_resources):
        """Time-indexed DP for each resource type considering urgency + wait time."""
        if len(patients) == 0:
            return [], 0.0

        resource_free_time = [0] * num_resources
        assignments, total_urgency = [], 0.0

        # Initially compute priority score
        patients['priority_score'] = patients['urgency_score']  # initial

        patient_indices = list(range(len(patients)))

        while patient_indices:
            # Sort patients dynamically by priority (urgency + alpha * expected wait)
            patient_indices.sort(key=lambda i: -patients.loc[i, 'priority_score'])
            patient_idx = patient_indices.pop(0)
            patient = patients.loc[patient_idx]

            arrival_slot = patient['arrival_slot']
            duration_slots = patient['duration_slots']
            urgency = patient['urgency_score']

            best_resource = None
            best_start_slot = None

            for res_id in range(num_resources):
                earliest_start = max(resource_free_time[res_id], arrival_slot)
                end_slot = earliest_start + duration_slots

                if end_slot <= self.num_slots:
                    if best_resource is None or earliest_start < best_start_slot:
                        best_resource = res_id
                        best_start_slot = earliest_start

            if best_resource is not None:
                assignments.append({
                    'patient_id': patient['patient_id'],
                    'start_slot': best_start_slot,
                    'end_slot': best_start_slot + duration_slots,
                    'resource_id': best_resource,
                    'urgency': urgency,
                    'duration_minutes': patient['treatment_duration'],
                })
                resource_free_time[best_resource] = best_start_slot + duration_slots
                total_urgency += urgency

                # Update priority scores of remaining patients based on their expected wait
                for i in patient_indices:
                    p = patients.loc[i]
                    # estimated wait = max(0, earliest free slot across resources - arrival slot)
                    est_wait = max(0, min(resource_free_time) - p['arrival_slot'])
                    patients.at[i, 'priority_score'] = p['urgency_score'] + self.alpha * est_wait

        return assignments, total_urgency

    def _calculate_metrics(self, df, assignments, sim_start, sim_end, total_urgency):
        """
        Compute performance metrics.
        MODIFIED: Only includes the standard metrics to match the Greedy output.
        """
        assigned_ids = {a['patient_id'] for a in assignments}
        df['assigned'] = df['patient_id'].isin(assigned_ids)

        wait_times = []
        treatment_minutes = {'Doctor': 0, 'ICU': 0}

        for a in assignments:
            p = df[df['patient_id'] == a['patient_id']].iloc[0]
            start_slot = int(a['start_slot'])
            start_time = sim_start + timedelta(minutes=start_slot * int(self.slot_minutes))
            wait = (start_time - p['arrival_time']).total_seconds() / 60
            wait_times.append(max(0, wait))
            treatment_minutes[p['resource_type']] += float(a['duration_minutes'])

        total_capacity = {
            'Doctor': self.num_doctors * self.total_time_hours * 60,
            'ICU': self.num_icu * self.total_time_hours * 60,
        }

        # Calculate Utilization
        util_doctor = (treatment_minutes['Doctor'] / total_capacity['Doctor']) * 100
        util_icu = (treatment_minutes['ICU'] / total_capacity['ICU']) * 100
        avg_util = (util_doctor + util_icu) / 2

        assigned_df = df[df['assigned']]
        waiting_df = df[~df['assigned']]

        return {
            'patients_assigned': int(len(assigned_df)),
            'patients_waiting': int(len(waiting_df)),
            'avg_wait_time': round(np.mean(wait_times), 2) if wait_times else 0,
            'total_wait_time': int(sum(wait_times)),
            'utilization_rate': round(avg_util, 2),
            'total_urgency_served': round(total_urgency, 2), # Retaining this but it will be labeled N/A in the print output below
        }


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    start = time.time()

    # --- Data Setup ---
    try:
        df = pd.read_csv("patient_data.csv")
        print("Loaded dataset successfully!\n")
    except FileNotFoundError:
        print("Error: 'patient_data.csv' not found. Creating a synthetic dataset for demonstration.")
        num_patients = 3000
        start_date = pd.to_datetime('2025-01-01 08:00:00')
        
        data = {
            'patient_id': range(1, num_patients + 1),
            'arrival_time': [start_date + timedelta(minutes=int(np.random.exponential(1.5))) 
                             for _ in range(num_patients)],
            'urgency_score': np.random.randint(1, 21, num_patients), 
            'treatment_duration': np.random.randint(10, 121, num_patients),
            'resource_type': np.random.choice(['Doctor', 'ICU'], num_patients, p=[0.75, 0.25])
        }
        df = pd.DataFrame(data)
        df = df.sort_values('arrival_time').reset_index(drop=True)
        print("Using synthetic data for simulation.")
    # --------------------------

    print("=" * 60)
    print("TIME-INDEXED DP (Urgency + Wait-Time Priority)")
    print("=" * 60)

    # Use the same parameters as the Greedy simulation for comparison
    allocator = TimeIndexedDP(num_doctors=100, num_icu=50, total_time_hours=50, slot_minutes=15, alpha=0.3)
    metrics, _ = allocator.allocate_resources(df.copy()) # Use a copy

    # Print only the required metrics to match the Greedy output standard
    print("\n=== DP RESULTS (Matching Greedy Output) ===")
    
    # Define the order and format to match the previous Greedy output
    output_metrics = {
        "patients_assigned": metrics['patients_assigned'],
        "patients_waiting": metrics['patients_waiting'],
        "avg_wait_time": metrics['avg_wait_time'],
        "total_wait_time": metrics['total_wait_time'],
        "utilization_rate": metrics['utilization_rate'],
        
    }

    for k, v in output_metrics.items():
        print(f"{k:<25}: {v}")

    end = time.time()
    elapsed = round(end - start, 2)
    print(f"\nExecution Time (s): {elapsed}")
    print("Approx. Time Complexity: O(N^2 * R) due to sorting inside while loop")