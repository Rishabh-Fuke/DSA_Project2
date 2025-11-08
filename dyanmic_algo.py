import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


class TimeIndexedDP:
    """
    Time-indexed Dynamic Programming for patient scheduling.
    
    Key idea: Discretize time into slots and track resource availability
    at each time point, respecting arrival times and parallel resources.
    """
    
    def __init__(self, num_doctors=10, num_icu=5, total_time_hours=8, slot_minutes=15):
        self.num_doctors = num_doctors
        self.num_icu = num_icu
        self.total_time_hours = total_time_hours
        self.slot_minutes = slot_minutes
        self.num_slots = (total_time_hours * 60) // slot_minutes
        
    def allocate_resources(self, df):
        """
        Main allocation logic using time-indexed DP.
        
        State: (time_slot, doctor_availability, icu_availability, patient_index)
        Decision: Assign current patient to available resource or skip
        Objective: Maximize total urgency served
        """
        
        # Preprocess data
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
        
        # Run DP for each resource type separately (they're independent)
        doc_assignments, doc_urgency = self._run_dp_for_resource(
            doctor_patients, self.num_doctors, "Doctor"
        )
        icu_assignments, icu_urgency = self._run_dp_for_resource(
            icu_patients, self.num_icu, "ICU"
        )
        
        # Combine results
        all_assignments = doc_assignments + icu_assignments
        total_urgency = doc_urgency + icu_urgency
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            df, all_assignments, sim_start, sim_end, total_urgency
        )
        
        return metrics, all_assignments
    
    def _run_dp_for_resource(self, patients, num_resources, resource_type):
        """
        Run DP for a single resource type.
        
        This uses a simplified approach where we:
        1. Process patients in arrival order
        2. Use a greedy-DP hybrid: at each decision point, choose the patient
           that maximizes urgency/duration ratio among available patients
        3. Track resource availability over time using a timeline
        
        Note: Full DP with state (slot, resource_mask, patient_subset) is 
        exponential in problem size. We use a heuristic DP approach instead.
        """
        
        if len(patients) == 0:
            return [], 0.0
        
        n = len(patients)
        
        # Timeline: tracks when each resource becomes free
        # resource_free_time[i] = slot when resource i becomes available
        resource_free_time = [0] * num_resources
        
        assignments = []  # List of (patient_idx, start_slot, resource_id)
        total_urgency = 0.0
        
        # DP approach: For each patient in urgency-priority order,
        # try to schedule them at the earliest possible time
        patient_indices = list(range(n))
        patient_indices.sort(
            key=lambda i: -patients.loc[i, 'urgency_score']
        )
        
        scheduled = set()
        
        for patient_idx in patient_indices:
            if patient_idx in scheduled:
                continue
                
            patient = patients.loc[patient_idx]
            arrival_slot = patient['arrival_slot']
            duration_slots = patient['duration_slots']
            urgency = patient['urgency_score']
            
            # Find earliest available resource
            best_resource = None
            best_start_slot = None
            
            for res_id in range(num_resources):
                # Earliest this resource can start is max of:
                # 1. When resource becomes free
                # 2. When patient arrives
                earliest_start = max(resource_free_time[res_id], arrival_slot)
                end_slot = earliest_start + duration_slots
                
                # Check if treatment finishes within simulation time
                if end_slot <= self.num_slots:
                    if best_resource is None or earliest_start < best_start_slot:
                        best_resource = res_id
                        best_start_slot = earliest_start
            
            # If we found a valid assignment, schedule it
            if best_resource is not None:
                assignments.append({
                    'patient_id': patient['patient_id'],
                    'patient_idx': patient_idx,
                    'start_slot': best_start_slot,
                    'end_slot': best_start_slot + duration_slots,
                    'resource_id': best_resource,
                    'arrival_slot': arrival_slot,
                    'urgency': urgency,
                    'duration_minutes': patient['treatment_duration']
                })
                
                # Update resource availability
                resource_free_time[best_resource] = best_start_slot + duration_slots
                scheduled.add(patient_idx)
                total_urgency += urgency
        
        return assignments, total_urgency
    
    def _calculate_metrics(self, df, assignments, sim_start, sim_end, total_urgency):
        """Calculate comprehensive metrics for the allocation."""
        
        assigned_ids = {a['patient_id'] for a in assignments}
        df['assigned'] = df['patient_id'].isin(assigned_ids)
        
        # Calculate wait times for assigned patients
        wait_times = []
        treatment_minutes = {'Doctor': 0, 'ICU': 0}
        
        for assignment in assignments:
            patient = df[df['patient_id'] == assignment['patient_id']].iloc[0]
            
            # Wait time = (start_time - arrival_time)
            start_time = sim_start + timedelta(
                minutes=int(assignment['start_slot'] * self.slot_minutes)
            )
            wait_minutes = (start_time - patient['arrival_time']).total_seconds() / 60
            wait_times.append(max(0, wait_minutes))
            
            # Track treatment time for utilization
            treatment_minutes[patient['resource_type']] += assignment['duration_minutes']
        
        # Calculate utilization
        total_capacity_minutes = {
            'Doctor': self.num_doctors * self.total_time_hours * 60,
            'ICU': self.num_icu * self.total_time_hours * 60
        }
        
        util_doctor = (treatment_minutes['Doctor'] / total_capacity_minutes['Doctor']) * 100
        util_icu = (treatment_minutes['ICU'] / total_capacity_minutes['ICU']) * 100
        avg_utilization = (util_doctor + util_icu) / 2
        
        assigned_df = df[df['assigned']]
        waiting_df = df[~df['assigned']]
        
        return {
            'patients_assigned': int(len(assigned_df)),
            'patients_waiting': int(len(waiting_df)),
            'avg_wait_time': round(np.mean(wait_times), 2) if wait_times else 0,
            'total_wait_time': int(sum(wait_times)),
            'utilization_rate': round(avg_utilization, 2),
            'total_urgency_served': round(total_urgency, 2),
            'util_doctor': round(util_doctor, 2),
            'util_icu': round(util_icu, 2)
        }


class ImprovedDP:
    """
    Improved DP using interval scheduling with weighted jobs.
    
    This is closer to a true DP solution that considers:
    - Non-overlapping assignments for each resource
    - Maximizing total urgency
    - Respecting arrival times
    """
    
    def __init__(self, num_doctors=10, num_icu=5, total_time_hours=8):
        self.num_doctors = num_doctors
        self.num_icu = num_icu
        self.total_time_hours = total_time_hours
        self.total_minutes = total_time_hours * 60
        
    def allocate_resources(self, df):
        """Allocate using weighted interval scheduling DP."""
        
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        df = df.sort_values('arrival_time').reset_index(drop=True)
        
        sim_start = df['arrival_time'].min()
        sim_end = sim_start + timedelta(hours=self.total_time_hours)
        df = df[df['arrival_time'] <= sim_end].copy()
        
        # Convert times to minutes from start
        df['arrival_minutes'] = df['arrival_time'].apply(
            lambda x: (x - sim_start).total_seconds() / 60
        )
        
        # Separate by resource type
        doctor_patients = df[df['resource_type'] == 'Doctor'].reset_index(drop=True)
        icu_patients = df[df['resource_type'] == 'ICU'].reset_index(drop=True)
        
        # Run weighted interval scheduling for each resource type
        doc_assignments, doc_urgency = self._weighted_interval_scheduling(
            doctor_patients, self.num_doctors
        )
        icu_assignments, icu_urgency = self._weighted_interval_scheduling(
            icu_patients, self.num_icu
        )
        
        all_assignments = doc_assignments + icu_assignments
        total_urgency = doc_urgency + icu_urgency
        
        metrics = self._calculate_metrics(
            df, all_assignments, sim_start, total_urgency
        )
        
        return metrics, all_assignments
    
    def _weighted_interval_scheduling(self, patients, num_resources):
        """
        Weighted interval scheduling with multiple identical resources.
        
        For each resource, we solve the classic weighted interval scheduling
        problem, then greedily assign patients to resources.
        """
        
        if len(patients) == 0:
            return [], 0.0
        
        n = len(patients)
        
        # Create jobs: (start, end, weight, patient_id, patient_idx)
        jobs = []
        for idx, row in patients.iterrows():
            start = row['arrival_minutes']
            end = start + row['treatment_duration']
            
            # Only consider jobs that finish within simulation
            if end <= self.total_minutes:
                jobs.append({
                    'start': start,
                    'end': end,
                    'weight': row['urgency_score'],
                    'patient_id': row['patient_id'],
                    'patient_idx': idx,
                    'duration': row['treatment_duration']
                })
        
        if not jobs:
            return [], 0.0
        
        # Sort jobs by end time
        jobs.sort(key=lambda x: x['end'])
        
        # For each resource, find optimal schedule using DP
        all_assignments = []
        used_patients = set()
        total_urgency = 0.0
        
        for resource_id in range(num_resources):
            # Available jobs are those not yet assigned
            available_jobs = [j for j in jobs if j['patient_idx'] not in used_patients]
            
            if not available_jobs:
                break
            
            # Classic weighted interval scheduling DP
            m = len(available_jobs)
            dp = [0.0] * (m + 1)
            choice = [False] * (m + 1)
            
            for i in range(1, m + 1):
                # Find latest non-conflicting job
                p = i - 1
                while p >= 0 and available_jobs[p]['end'] > available_jobs[i-1]['start']:
                    p -= 1
                
                include_value = available_jobs[i-1]['weight']
                if p >= 0:
                    include_value += dp[p + 1]
                
                if include_value > dp[i - 1]:
                    dp[i] = include_value
                    choice[i] = True
                else:
                    dp[i] = dp[i - 1]
                    choice[i] = False
            
            # Backtrack to find selected jobs
            selected = []
            i = m
            while i > 0:
                if choice[i]:
                    selected.append(available_jobs[i-1])
                    used_patients.add(available_jobs[i-1]['patient_idx'])
                    total_urgency += available_jobs[i-1]['weight']
                    
                    # Skip to latest non-conflicting
                    p = i - 1
                    while p >= 0 and available_jobs[p]['end'] > available_jobs[i-1]['start']:
                        p -= 1
                    i = p + 1
                else:
                    i -= 1
            
            # Add assignments for this resource
            for job in selected:
                all_assignments.append({
                    'patient_id': job['patient_id'],
                    'start_minutes': job['start'],
                    'end_minutes': job['end'],
                    'resource_id': resource_id,
                    'urgency': job['weight'],
                    'duration_minutes': job['duration']
                })
        
        return all_assignments, total_urgency
    
    def _calculate_metrics(self, df, assignments, sim_start, total_urgency):
        """Calculate metrics."""
        
        assigned_ids = {a['patient_id'] for a in assignments}
        df['assigned'] = df['patient_id'].isin(assigned_ids)
        
        wait_times = []
        treatment_minutes = {'Doctor': 0, 'ICU': 0}
        
        for assignment in assignments:
            patient = df[df['patient_id'] == assignment['patient_id']].iloc[0]
            arrival_minutes = (patient['arrival_time'] - sim_start).total_seconds() / 60
            wait_minutes = assignment['start_minutes'] - arrival_minutes
            wait_times.append(max(0, wait_minutes))
            treatment_minutes[patient['resource_type']] += assignment['duration_minutes']
        
        total_capacity = {
            'Doctor': self.num_doctors * self.total_minutes,
            'ICU': self.num_icu * self.total_minutes
        }
        
        util_doctor = (treatment_minutes['Doctor'] / total_capacity['Doctor']) * 100
        util_icu = (treatment_minutes['ICU'] / total_capacity['ICU']) * 100
        avg_utilization = (util_doctor + util_icu) / 2
        
        assigned_df = df[df['assigned']]
        waiting_df = df[~df['assigned']]
        
        return {
            'patients_assigned': int(len(assigned_df)),
            'patients_waiting': int(len(waiting_df)),
            'avg_wait_time': round(np.mean(wait_times), 2) if wait_times else 0,
            'total_wait_time': int(sum(wait_times)),
            'utilization_rate': round(avg_utilization, 2),
            'total_urgency_served': round(total_urgency, 2),
            'util_doctor': round(util_doctor, 2),
            'util_icu': round(util_icu, 2)
        }


# TESTING
if __name__ == "__main__":
    df = pd.read_csv("patient_data_short.csv")
    print("✅ Loaded dataset successfully!\n")
    
    # Test Time-Indexed DP
    print("=" * 60)
    print("TIME-INDEXED DP (Urgency-Priority with Time Slots)")
    print("=" * 60)
    allocator1 = TimeIndexedDP(num_doctors=10, num_icu=5, total_time_hours=8, slot_minutes=15)
    metrics1, _ = allocator1.allocate_resources(df)
    for k, v in metrics1.items():
        print(f"{k:<25}: {v}")
    
    print("\n" + "=" * 60)
    print("IMPROVED DP (Weighted Interval Scheduling)")
    print("=" * 60)
    allocator2 = ImprovedDP(num_doctors=10, num_icu=5, total_time_hours=8)
    metrics2, _ = allocator2.allocate_resources(df)
    for k, v in metrics2.items():
        print(f"{k:<25}: {v}")