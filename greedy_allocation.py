import pandas as pd
import numpy as np
from queue import PriorityQueue
from datetime import datetime, timedelta
import argparse

class ResourcePool:
    def __init__(self, num_doctors=10, num_icu=5):
        self.doctors = set(range(num_doctors))  # Available doctor IDs
        self.icu_beds = set(range(num_icu))    # Available ICU bed IDs
        self.doctor_assignments = {}  # Keep track of when each doctor will be free
        self.icu_assignments = {}     # Keep track of when each ICU bed will be free
    
    def get_next_available_resource(self, resource_type, current_time):
        if resource_type == "Doctor":
            assignments = self.doctor_assignments
            resources = self.doctors
        else:  # ICU
            assignments = self.icu_assignments
            resources = self.icu_beds
            
        # Find the resource that will be available soonest
        earliest_available = None
        earliest_time = None
        # Clean up assignments that have already freed by current_time
        freed = [r for r, t in assignments.items() if t <= current_time]
        for r in freed:
            del assignments[r]

        # First check if any resource is completely free now
        available_now = resources - set(assignments.keys())
        if available_now:
            return available_now.pop(), current_time
            
        # If no resource is completely free, find the one that will be free first
        for resource_id in resources:
            if resource_id in assignments:
                free_time = assignments[resource_id]
                if earliest_time is None or free_time < earliest_time:
                    earliest_time = free_time
                    earliest_available = resource_id
                    
        return earliest_available, earliest_time

    def assign_resource(self, resource_type, resource_id, start_time, duration):
        end_time = start_time + timedelta(minutes=int(duration))
        if resource_type == "Doctor":
            self.doctor_assignments[resource_id] = end_time
        else:  # ICU
            self.icu_assignments[resource_id] = end_time

class GreedyAllocator:
    def __init__(self, num_doctors=10, num_icu=5):
        self.resource_pool = ResourcePool(num_doctors, num_icu)
        self.waiting_times = {"Doctor": [], "ICU": []}
        self.resource_utilization = {"Doctor": [], "ICU": []}
        self.unassigned = {"Doctor": [], "ICU": []}
        self._counter = 0
        
    @staticmethod
    def analyze_capacity_requirements(df):
        """Analyze patient arrival patterns and suggest resource levels"""
        df = df.copy()
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        
        # Calculate metrics per resource type
        analysis = {}
        for resource in ['Doctor', 'ICU']:
            resource_df = df[df['resource_type'] == resource]
            if len(resource_df) == 0:
                continue
                
            # Calculate arrival rate (patients per hour)
            time_span = (resource_df['arrival_time'].max() - resource_df['arrival_time'].min()).total_seconds() / 3600
            arrival_rate = len(resource_df) / time_span if time_span > 0 else 0
            
            # Calculate average treatment time
            avg_treatment = resource_df['treatment_duration'].mean()
            
            # Calculate minimum resources needed using Little's Law
            # L = λW where L is avg patients, λ is arrival rate, W is service time
            min_resources = np.ceil(arrival_rate * (avg_treatment / 60))
            
            # Add 20% buffer for peak times and urgent cases
            recommended = int(np.ceil(min_resources * 1.2))
            
            analysis[resource] = {
                'arrival_rate_per_hour': arrival_rate,
                'avg_treatment_mins': avg_treatment,
                'min_resources_needed': int(min_resources),
                'recommended_resources': recommended,
                'total_patients': len(resource_df),
                'peak_parallel_treatments': int(np.ceil(min_resources * 1.5))  # For worst-case scenario
            }
            
        return analysis
        
    def priority_score(self, urgency, wait_time_minutes):
        """
        Calculate priority score based on urgency and waiting time.
        Urgency is weighted more heavily but waiting time is also considered
        to prevent starvation of lower-urgency patients.
        """
        wait_time_hours = wait_time_minutes / 60
        return urgency * 2 + wait_time_hours  # Urgency has higher weight

    def allocate_resources(self, df):
        # Convert arrival_time to datetime if it's not already
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        # Sort by arrival time to simulate timeline
        df = df.sort_values('arrival_time').reset_index(drop=True)
        
        # Create priority queue for each resource type
        doctor_queue = PriorityQueue()
        icu_queue = PriorityQueue()
        
        # Process patients in order of arrival time
        for _, row in df.iterrows():
            current_time = row['arrival_time']

            # Build a lightweight patient dict to avoid Series comparability issues
            patient = {
                'patient_id': int(row.get('patient_id', -1)),
                'urgency_score': float(row['urgency_score']),
                'arrival_time': row['arrival_time'],
                'treatment_duration': int(row['treatment_duration']),
                'resource_type': row['resource_type']
            }

            # Add patient to appropriate queue with negative priority (for max-heap behavior)
            self._counter += 1
            priority = -self.priority_score(patient['urgency_score'], 0)
            if patient['resource_type'] == "Doctor":
                doctor_queue.put((priority, self._counter, patient))
            else:  # ICU
                icu_queue.put((priority, self._counter, patient))

            # Attempt to allocate one patient from each queue (keeps runtime bounded)
            if not doctor_queue.empty():
                _, _, current_patient = doctor_queue.get()
                resource_id, available_time = self.resource_pool.get_next_available_resource("Doctor", current_time)

                # If there are no doctor resources at all, mark as unassigned
                if resource_id is None and available_time is None:
                    self.unassigned["Doctor"].append(current_patient['patient_id'])
                    continue

                # Calculate waiting time (non-negative)
                wait_time = max(0, (available_time - current_patient['arrival_time']).total_seconds() / 60)
                self.waiting_times["Doctor"].append(wait_time)

                # Assign the resource
                self.resource_pool.assign_resource("Doctor", resource_id, available_time,
                                                  current_patient['treatment_duration'])

                # Record utilization
                self.resource_utilization["Doctor"].append({
                    'patient_id': current_patient['patient_id'],
                    'resource_id': resource_id,
                    'start_time': available_time,
                    'end_time': available_time + timedelta(minutes=int(current_patient['treatment_duration'])),
                    'wait_time': wait_time,
                    'urgency_score': current_patient['urgency_score']
                })

            if not icu_queue.empty():
                _, _, current_patient = icu_queue.get()
                resource_id, available_time = self.resource_pool.get_next_available_resource("ICU", current_time)

                if resource_id is None and available_time is None:
                    self.unassigned["ICU"].append(current_patient['patient_id'])
                    continue

                wait_time = max(0, (available_time - current_patient['arrival_time']).total_seconds() / 60)
                self.waiting_times["ICU"].append(wait_time)

                self.resource_pool.assign_resource("ICU", resource_id, available_time,
                                                  current_patient['treatment_duration'])

                self.resource_utilization["ICU"].append({
                    'patient_id': current_patient['patient_id'],
                    'resource_id': resource_id,
                    'start_time': available_time,
                    'end_time': available_time + timedelta(minutes=int(current_patient['treatment_duration'])),
                    'wait_time': wait_time,
                    'urgency_score': current_patient['urgency_score']
                })
    
    def get_metrics(self):
        """Return performance metrics for the allocation"""
        metrics = {}
        total_waiting = []
        total_processed = 0
        
        # Calculate waiting time metrics per resource type
        for resource_type in ["Doctor", "ICU"]:
            waits = self.waiting_times[resource_type]
            if waits:
                metrics[f'{resource_type}_avg_wait'] = float(np.mean(waits))
                metrics[f'{resource_type}_max_wait'] = float(np.max(waits))
                metrics[f'{resource_type}_min_wait'] = float(np.min(waits))
                metrics[f'{resource_type}_patients_processed'] = len(waits)
                metrics[f'{resource_type}_unassigned'] = len(self.unassigned[resource_type])
                total_waiting.extend(waits)
                total_processed += len(waits)
            else:
                metrics[f'{resource_type}_avg_wait'] = 0.0
                metrics[f'{resource_type}_max_wait'] = 0.0
                metrics[f'{resource_type}_min_wait'] = 0.0
                metrics[f'{resource_type}_patients_processed'] = 0
                metrics[f'{resource_type}_unassigned'] = len(self.unassigned[resource_type])
        
        # Overall metrics
        if total_waiting:
            metrics.update({
                'average_wait_time': float(np.mean(total_waiting)),
                'max_wait_time': float(np.max(total_waiting)),
                'min_wait_time': float(np.min(total_waiting)),
                'total_patients_processed': total_processed
            })
        else:
            metrics.update({
                'average_wait_time': 0.0,
                'max_wait_time': 0.0,
                'min_wait_time': 0.0,
                'total_patients_processed': 0
            })

        # Calculate resource utilization
        for resource_type in ["Doctor", "ICU"]:
            total_time = timedelta()
            starts = []
            ends = []
            for allocation in self.resource_utilization[resource_type]:
                duration = allocation['end_time'] - allocation['start_time']
                total_time += duration
                starts.append(allocation['start_time'])
                ends.append(allocation['end_time'])

            # Calculate utilization percentage over the observed time span
            if resource_type == "Doctor":
                num_resources = len(self.resource_pool.doctors)
            else:
                num_resources = len(self.resource_pool.icu_beds)

            if len(starts) == 0 or num_resources == 0:
                metrics[f'{resource_type}_utilization'] = 0.0
            else:
                span_start = min(starts)
                span_end = max(ends)
                time_span = span_end - span_start
                # Avoid zero division
                if time_span.total_seconds() <= 0:
                    time_span = timedelta(minutes=1)

                total_possible_time = time_span * num_resources
                metrics[f'{resource_type}_utilization'] = (total_time / total_possible_time) * 100

        return metrics

def main():
    parser = argparse.ArgumentParser(description='Greedy Hospital Resource Allocator')
    parser.add_argument('--doctors', type=int, default=10, help='Number of doctors')
    parser.add_argument('--icu', type=int, default=5, help='Number of ICU beds')
    parser.add_argument('--sample', type=int, default=None, help='Sample N rows from dataset for quick runs')
    parser.add_argument('--csv', type=str, default='patient_data.csv', help='Path to patient CSV')
    args = parser.parse_args()

    # Read the patient data
    df = pd.read_csv(args.csv)
    if args.sample is not None:
        df = df.sample(n=args.sample) if args.sample <= len(df) else df
        
    # Analyze capacity requirements
    print("\nCapacity Analysis:")
    print("-" * 50)
    analysis = GreedyAllocator.analyze_capacity_requirements(df)
    for resource, stats in analysis.items():
        print(f"\n{resource} Requirements:")
        print(f"Arrival Rate: {stats['arrival_rate_per_hour']:.1f} patients/hour")
        print(f"Average Treatment: {stats['avg_treatment_mins']:.1f} minutes")
        print(f"Minimum Resources: {stats['min_resources_needed']} {resource}s")
        print(f"Recommended (with 20% buffer): {stats['recommended_resources']} {resource}s")
        print(f"Peak Parallel Load: {stats['peak_parallel_treatments']} {resource}s")
        
    print("\nCurrent Resource Levels:")
    print(f"Doctors: {args.doctors} (vs recommended {analysis.get('Doctor', {}).get('recommended_resources', 'N/A')})")
    print(f"ICU Beds: {args.icu} (vs recommended {analysis.get('ICU', {}).get('recommended_resources', 'N/A')})")

    # Initialize the greedy allocator with specified resource counts
    allocator = GreedyAllocator(num_doctors=args.doctors, num_icu=args.icu)

    # Run the allocation
    allocator.allocate_resources(df)

    # Get and print metrics
    metrics = allocator.get_metrics()
    print("\nAllocation Results:")
    print("-" * 50)
    
    print("\nOverall Metrics:")
    print(f"Total Patients Processed: {metrics['total_patients_processed']}")
    print(f"Average Wait Time: {metrics['average_wait_time']:.2f} minutes")
    print(f"Maximum Wait Time: {metrics['max_wait_time']:.2f} minutes")
    
    for resource in ["Doctor", "ICU"]:
        print(f"\n{resource} Metrics:")
        print(f"Patients Processed: {metrics[f'{resource}_patients_processed']}")
        print(f"Unassigned Patients: {metrics[f'{resource}_unassigned']}")
        print(f"Average Wait: {metrics[f'{resource}_avg_wait']:.2f} minutes")
        print(f"Maximum Wait: {metrics[f'{resource}_max_wait']:.2f} minutes")
        print(f"Utilization: {metrics.get(f'{resource}_utilization', 0.0):.2f}%")

if __name__ == "__main__":
    main()