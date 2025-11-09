import pandas as pd
import numpy as np
from queue import PriorityQueue
from datetime import datetime, timedelta
import warnings
import time # Import time module for execution time tracking
warnings.filterwarnings("ignore")


class ResourcePool:
    def __init__(self, num_doctors=10, num_icu=5):
        self.doctors = set(range(num_doctors))
        self.icu_beds = set(range(num_icu))
        self.doctor_assignments = {}
        self.icu_assignments = {}

    def get_next_available_resource(self, resource_type, current_time):
        if resource_type == "Doctor":
            assignments = self.doctor_assignments
            resources = self.doctors
        else:
            assignments = self.icu_assignments
            resources = self.icu_beds

        # Release completed assignments (O(R))
        freed = [r for r, t in assignments.items() if t <= current_time]
        for r in freed:
            del assignments[r]

        available_now = resources - set(assignments.keys())
        if available_now:
            return available_now.pop(), current_time

        # Find earliest future availability (O(R))
        if not assignments:
             return None, None
             
        earliest_time = None
        earliest_resource = None
        for r, t in assignments.items():
            if earliest_time is None or t < earliest_time:
                earliest_time = t
                earliest_resource = r
        return earliest_resource, earliest_time

    def assign_resource(self, resource_type, resource_id, start_time, duration):
        end_time = start_time + timedelta(minutes=int(duration))
        if resource_type == "Doctor":
            self.doctor_assignments[resource_id] = end_time
        else:
            self.icu_assignments[resource_id] = end_time


class GreedyAllocator:
    def __init__(self, num_doctors=10, num_icu=5, total_time_hours=8):
        self.num_doctors = num_doctors
        self.num_icu = num_icu
        self.total_time_minutes = total_time_hours * 60
        self.simulation_end = None
        self.resource_pool = ResourcePool(num_doctors, num_icu)
        self.waiting_times = {"Doctor": [], "ICU": []}
        self.unassigned = {"Doctor": [], "ICU": []}
        self.treatment_times = {"Doctor": 0, "ICU": 0}

    @staticmethod
    def priority_score(urgency, wait_time_minutes):
        """Priority based ONLY on urgency."""
        return urgency * 2 

    def allocate_resources(self, df):
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        df = df.sort_values('arrival_time').reset_index(drop=True)

        doctor_queue = PriorityQueue()
        icu_queue = PriorityQueue()

        sim_start_time = df['arrival_time'].min()
        self.simulation_end = sim_start_time + timedelta(hours=50)
        df = df[df['arrival_time'] <= self.simulation_end].copy()

        current_time = sim_start_time
        N = len(df) # Total number of patients

        # Process patients in arrival order (Outer loop runs N times)
        for _, row in df.iterrows():
            patient_time = row['arrival_time']
            
            if patient_time > current_time:
                current_time = patient_time

            patient = {
                'patient_id': int(row['patient_id']),
                'urgency_score': float(row['urgency_score']),
                'arrival_time': patient_time,
                'treatment_duration': int(row['treatment_duration']),
                'resource_type': row['resource_type'],
                'wait_start': patient_time
            }
            priority = -self.priority_score(patient['urgency_score'], 0)
            
            # Queue insertion is O(log Q), where Q is queue size (<= N).
            if patient['resource_type'] == "Doctor":
                doctor_queue.put((priority, patient))
            else:
                icu_queue.put((priority, patient))

            # Try to assign from each queue (Inner logic runs 2 times)
            for q, rtype in [(doctor_queue, "Doctor"), (icu_queue, "ICU")]:
                if q.empty():
                    continue
                
                # Dequeue is O(log Q)
                _, p = q.get()

                # get_next_available_resource is O(R) (R = number of resources)
                resource_id, available_time = self.resource_pool.get_next_available_resource(rtype, current_time)
                
                if resource_id is None:
                    self.unassigned[rtype].append(p['patient_id'])
                    continue

                start_time = max(available_time, p['arrival_time'])
                end_time = start_time + timedelta(minutes=p['treatment_duration'])

                if end_time > self.simulation_end:
                    self.unassigned[rtype].append(p['patient_id'])
                    continue

                self.waiting_times[rtype].append((start_time - p['arrival_time']).total_seconds() / 60)
                self.treatment_times[rtype] += p['treatment_duration']
                self.resource_pool.assign_resource(rtype, resource_id, start_time, p['treatment_duration'])
                
        # Final cleanup O(R)
        for rtype in ["Doctor", "ICU"]:
            assignments = (self.resource_pool.doctor_assignments if rtype == "Doctor"
                           else self.resource_pool.icu_assignments)
            
            freed = [r for r, t in assignments.items() if t <= self.simulation_end]
            for r in freed:
                del assignments[r]

    def get_metrics(self):
        total_assigned = len(self.waiting_times["Doctor"]) + len(self.waiting_times["ICU"])
        total_waiting = len(self.unassigned["Doctor"]) + len(self.unassigned["ICU"])
        all_wait_times = self.waiting_times["Doctor"] + self.waiting_times["ICU"]
        avg_wait = np.mean(all_wait_times) if all_wait_times else 0
        total_wait = np.sum(all_wait_times)

        sim_duration_minutes = 3000 
        cap_doc = self.num_doctors * sim_duration_minutes
        cap_icu = self.num_icu * sim_duration_minutes
        
        util_doc = min(100.0, (self.treatment_times["Doctor"] / cap_doc) * 100) if cap_doc > 0 else 0
        util_icu = min(100.0, (self.treatment_times["ICU"] / cap_icu) * 100) if cap_icu > 0 else 0
        utilization_rate = (util_doc + util_icu) / 2
        
        return {
            "patients_assigned": int(total_assigned),
            "patients_waiting": int(total_waiting),
            "avg_wait_time": round(avg_wait, 2),
            "total_wait_time": int(total_wait),
            "utilization_rate": round(utilization_rate, 2),
            "total_urgency_served": "N/A"
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
        # Using a large patient count (e.g., 3000) to ensure a measurable execution time
        num_patients = 3000 
        start_date = datetime(2025, 1, 1, 8, 0, 0)
        
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

    # Running the simulation with fixed parameters
    allocator = GreedyAllocator(num_doctors=100, num_icu=50, total_time_hours=50)
    allocator.allocate_resources(df.copy())
    metrics = allocator.get_metrics()

    end = time.time()
    elapsed = round(end - start, 2)

    print("\n=== GREEDY RESULTS (Urgency ONLY Priority) ===")
    for k, v in metrics.items():
        print(f"{k:<35}: {v}")
    
    # --- OUTPUT REQUESTED ---
    print(f"\nExecution Time (s): {elapsed}")
    print("Approx. Time Complexity: O(N * (log N + R))")