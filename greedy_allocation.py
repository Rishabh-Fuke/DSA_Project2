import pandas as pd
import numpy as np
from queue import PriorityQueue
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# global simulation parameters
SIM_NUM_DOCTORS = 100
SIM_NUM_ICU = 50
SIM_TOTAL_TIME_HOURS = 1660


class ResourcePool:
    # manages resource assignments and availability
    def __init__(self, num_doctors=SIM_NUM_DOCTORS, num_icu=SIM_NUM_ICU):
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

        # releases completed assignments
        freed = [r for r, t in assignments.items() if t <= current_time]
        for r in freed:
            del assignments[r]

        available_now = resources - set(assignments.keys())
        if available_now:
            return available_now.pop(), current_time

        # finds earliest future availability
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
    # allocates resources based on urgency score only
    def __init__(self, num_doctors, num_icu, total_time_hours):
        self.num_doctors = num_doctors
        self.num_icu = num_icu
        self.total_time_hours = total_time_hours
        self.simulation_end = None
        self.resource_pool = ResourcePool(num_doctors, num_icu)
        self.waiting_times = {"Doctor": [], "ICU": []}
        self.unassigned = {"Doctor": [], "ICU": []}
        self.treatment_times = {"Doctor": 0, "ICU": 0}

    @staticmethod
    def priority_score(urgency, wait_time_minutes):
        return urgency * 2 

    def allocate_resources(self, df):
        # main allocation simulation.
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        df = df.sort_values('arrival_time').reset_index(drop=True)

        doctor_queue = PriorityQueue()
        icu_queue = PriorityQueue()

        sim_start_time = df['arrival_time'].min()
        self.simulation_end = sim_start_time + timedelta(hours=self.total_time_hours)
        df = df[df['arrival_time'] <= self.simulation_end].copy()

        current_time = sim_start_time

        # process patients in arrival order
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
            
            if patient['resource_type'] == "Doctor":
                doctor_queue.put((priority, patient))
            else:
                icu_queue.put((priority, patient))

            # try to assign the highest priority patient from each queue
            for q, rtype in [(doctor_queue, "Doctor"), (icu_queue, "ICU")]:
                if q.empty():
                    continue
                
                _, p = q.get()

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
                
        for rtype in ["Doctor", "ICU"]:
            assignments = (self.resource_pool.doctor_assignments if rtype == "Doctor"
                           else self.resource_pool.icu_assignments)
            
            freed = [r for r, t in assignments.items() if t <= self.simulation_end]
            for r in freed:
                del assignments[r]

    def get_metrics(self):
        # calculate assigned and waiting patients
        total_assigned = len(self.waiting_times["Doctor"]) + len(self.waiting_times["ICU"])
        total_waiting = len(self.unassigned["Doctor"]) + len(self.unassigned["ICU"])
        all_wait_times = self.waiting_times["Doctor"] + self.waiting_times["ICU"]
        avg_wait = np.mean(all_wait_times) if all_wait_times else 0
        total_wait = np.sum(all_wait_times)

        # utilization calculation: Fixed duration
        sim_duration_minutes = self.total_time_hours * 60 
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


if __name__ == "__main__":
    start = time.time()

    try:
        df = pd.read_csv("patient_data.csv")
        print("Loaded dataset successfully!\n")
    except FileNotFoundError:
        print("Error: 'patient_data.csv' not found. Creating a synthetic dataset for demonstration.")
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

    print("=" * 60)
    print("GREEDY RESULTS (Urgency ONLY Priority)")
    print("=" * 60)
    
    allocator_greedy = GreedyAllocator(
        num_doctors=SIM_NUM_DOCTORS,
        num_icu=SIM_NUM_ICU,
        total_time_hours=SIM_TOTAL_TIME_HOURS
    )
    allocator_greedy.allocate_resources(df.copy())
    metrics_greedy = allocator_greedy.get_metrics()
    
    end = time.time()
    elapsed = round(end - start, 2)

    output_metrics_greedy = {
        "patients_assigned": metrics_greedy['patients_assigned'],
        "patients_waiting": metrics_greedy['patients_waiting'],
        "avg_wait_time": metrics_greedy['avg_wait_time'],
        "total_wait_time": metrics_greedy['total_wait_time'],
        "utilization_rate": metrics_greedy['utilization_rate'],
        
    }

    for k, v in output_metrics_greedy.items():
        print(f"{k:<35}: {v}")

    print(f"\nExecution Time (s): {elapsed}")
    print("Approx. Time Complexity: O(N * (log N + R))")