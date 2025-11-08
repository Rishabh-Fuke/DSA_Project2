import pandas as pd
import numpy as np
from queue import PriorityQueue
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class ResourcePool:
    def __init__(self, num_doctors=10, num_icu=5):
        self.doctors = set(range(num_doctors))
        self.icu_beds = set(range(num_icu))
        self.doctor_assignments = {}  # resource_id -> end_time
        self.icu_assignments = {}

    def get_next_available_resource(self, resource_type, current_time):
        if resource_type == "Doctor":
            assignments = self.doctor_assignments
            resources = self.doctors
        else:
            assignments = self.icu_assignments
            resources = self.icu_beds

        # Release completed assignments
        freed = [r for r, t in assignments.items() if t <= current_time]
        for r in freed:
            del assignments[r]

        available_now = resources - set(assignments.keys())
        if available_now:
            return available_now.pop(), current_time

        # Find earliest future availability
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
        self.treatment_times = {"Doctor": 0, "ICU": 0}  # Total busy minutes

    @staticmethod
    def priority_score(urgency, wait_time_minutes):
        return urgency * 2 + (wait_time_minutes / 60)

    def allocate_resources(self, df):
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        df = df.sort_values('arrival_time').reset_index(drop=True)

        doctor_queue = PriorityQueue()
        icu_queue = PriorityQueue()

        self.simulation_end = df['arrival_time'].min() + timedelta(hours=8)
        df = df[df['arrival_time'] <= self.simulation_end].copy()

        current_time = df['arrival_time'].min()

        # Process patients in arrival order
        for _, row in df.iterrows():
            patient_time = row['arrival_time']
            if patient_time > current_time:
                current_time = patient_time

            # Add patient to correct queue
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

            # Try to assign from each queue
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

                # CRITICAL: Do not start if treatment ends after simulation
                if end_time > self.simulation_end:
                    self.unassigned[rtype].append(p['patient_id'])
                    # Re-queue if possible? Or just reject
                    continue

                wait_time = (start_time - p['arrival_time']).total_seconds() / 60
                self.waiting_times[rtype].append(wait_time)
                self.treatment_times[rtype] += p['treatment_duration']
                self.resource_pool.assign_resource(rtype, resource_id, start_time, p['treatment_duration'])

        # Final cleanup: release all
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

        # Correct utilization: treatment time / capacity
        cap_doc = self.num_doctors * self.total_time_minutes
        cap_icu = self.num_icu * self.total_time_minutes
        util_doc = min(100.0, (self.treatment_times["Doctor"] / cap_doc) * 100)
        util_icu = min(100.0, (self.treatment_times["ICU"] / cap_icu) * 100)
        utilization_rate = (util_doc + util_icu) / 2

        return {
            "patients_assigned": int(total_assigned),
            "patients_waiting": int(total_waiting),
            "avg_wait_time": round(avg_wait, 2),
            "total_wait_time": int(total_wait),
            "utilization_rate": round(utilization_rate, 2),
            "total_urgency_served": "N/A"
        }


# RUN 
if __name__ == "__main__":
    df = pd.read_csv("patient_data_short.csv")
    print("Loaded dataset successfully!")

    allocator = GreedyAllocator(num_doctors=10, num_icu=5, total_time_hours=8)
    allocator.allocate_resources(df)
    metrics = allocator.get_metrics()

    print("\n=== GREEDY RESULTS (Fixed) ===")
    for k, v in metrics.items():
        print(f"{k:<25}: {v}")