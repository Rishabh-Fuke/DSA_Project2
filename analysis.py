import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from greedy_allocation import GreedyAllocator, SIM_NUM_DOCTORS, SIM_NUM_ICU, SIM_TOTAL_TIME_HOURS
from dynamic_algo import TimeIndexedDP

BATCH_SIZE = 1000
TOTAL_PATIENTS = 100000

def load_or_create_dataset(num_patients):
    # Try to load existing dataset
    try:
        df = pd.read_csv("patient_data.csv")
        print(f"Loaded dataset with {len(df)} patients\n")
        return df.head(num_patients)
    except FileNotFoundError:
        print(f"Creating synthetic dataset with {num_patients} patients...")
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
        df.to_csv("patient_data.csv", index=False)
        print(f"Dataset created and saved to patient_data.csv\n")
        return df

def run_greedy_algorithm(df):
    # run greedy allocator
    print("=" * 70)
    print("Running GREEDY Algorithm (Full Dataset)")
    print("=" * 70)
    
    start = time.time()
    allocator = GreedyAllocator(
        num_doctors=SIM_NUM_DOCTORS,
        num_icu=SIM_NUM_ICU,
        total_time_hours=SIM_TOTAL_TIME_HOURS
    )
    allocator.allocate_resources(df.copy())
    metrics = allocator.get_metrics()
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f}s")
    print(f"Patients Assigned: {metrics['patients_assigned']}")
    print(f"Patients Waiting: {metrics['patients_waiting']}")
    print(f"Avg Wait Time: {metrics['avg_wait_time']:.2f} min")
    print(f"Utilization Rate: {metrics['utilization_rate']:.2f}%\n")
    
    return metrics

def run_dynamic_cumulative(df, batch_num, total_batches):
    # run the dynamic allocator
    num_patients = batch_num * BATCH_SIZE
    print(f"Running DYNAMIC Algorithm - Batch {batch_num}/{total_batches} "
          f"(Patients 1 to {num_patients})")
    
    start = time.time()
    allocator = TimeIndexedDP(
        num_doctors=SIM_NUM_DOCTORS,
        num_icu=SIM_NUM_ICU,
        total_time_hours=SIM_TOTAL_TIME_HOURS,
        slot_minutes=15,
        alpha=0.3
    )
    metrics, _ = allocator.allocate_resources(df.copy())
    elapsed = time.time() - start
    
    print(f"Batch {batch_num} completed in {elapsed:.2f}s")
    print(f"Patients Assigned: {metrics['patients_assigned']}")
    print(f"Patients Waiting: {metrics['patients_waiting']}")
    print(f"Avg Wait Time: {metrics['avg_wait_time']:.2f} min")
    print(f"Utilization Rate: {metrics['utilization_rate']:.2f}%\n")
    
    return metrics

def plot_comparison(greedy_metrics, dynamic_metrics, batch_num, total_batches, greedy_only=False):
    # create plots    
    if greedy_only:
        metrics_to_plot = {
            'Patients Assigned': greedy_metrics['patients_assigned'],
            'Patients Waiting': greedy_metrics['patients_waiting'],
            'Avg Wait Time (min)': greedy_metrics['avg_wait_time'],
            'Utilization Rate (%)': greedy_metrics['utilization_rate']
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Greedy Algorithm Results (Full Dataset)', 
                     fontsize=16, fontweight='bold')
        
        for idx, (metric_name, value) in enumerate(metrics_to_plot.items()):
            ax = axes[idx // 2, idx % 2]
            
            bar = ax.bar([0], [value], color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5, width=0.5)
            
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_xticks([0])
            ax.set_xticklabels(['Greedy\n(Full Dataset)'], fontsize=10)
            ax.set_title(metric_name, fontsize=12, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            height = bar[0].get_height()
            ax.text(bar[0].get_x() + bar[0].get_width()/2., height,
                   f'{value:.1f}' if isinstance(value, float) else f'{value}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = 'greedy_results.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved greedy results chart: {filename}\n")
        plt.close()
        return
    
    dynamic_avg_wait = dynamic_metrics['avg_wait_time']
    
    metrics_to_plot = {
        'Patients Assigned': (greedy_metrics['patients_assigned'], 
                             dynamic_metrics['patients_assigned']),
        'Patients Waiting': (greedy_metrics['patients_waiting'], 
                            dynamic_metrics['patients_waiting']),
        'Avg Wait Time (min)': (greedy_metrics['avg_wait_time'], 
                               dynamic_avg_wait),
        'Utilization Rate (%)': (greedy_metrics['utilization_rate'], 
                                dynamic_metrics['utilization_rate'])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    num_patients = batch_num * BATCH_SIZE
    fig.suptitle(f'Algorithm Comparison: Greedy (Full) vs Dynamic (0-{num_patients} patients)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c']
    
    for idx, (metric_name, (greedy_val, dynamic_val)) in enumerate(metrics_to_plot.items()):
        ax = axes[idx // 2, idx % 2]
        
        x_pos = np.arange(2)
        values = [greedy_val, dynamic_val]
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Greedy\n(Full Dataset)', f'Dynamic\n(0-{num_patients})'], fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}' if isinstance(val, float) else f'{val}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    filename = f'comparison_batch_{batch_num:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison chart: {filename}\n")
    plt.close()

def main():
    print("\n" + "=" * 70)
    print("CUMULATIVE ALGORITHM COMPARISON SYSTEM")
    print("=" * 70 + "\n")
    
    df_full = load_or_create_dataset(TOTAL_PATIENTS)
    
    greedy_metrics = run_greedy_algorithm(df_full)
    plot_comparison(greedy_metrics, None, None, None, greedy_only=True)
    
    num_batches = TOTAL_PATIENTS // BATCH_SIZE
    
    print("=" * 70)
    print(f"Running DYNAMIC Algorithm cumulatively in {num_batches} batches")
    print("=" * 70 + "\n")
    
    for batch_num in range(1, num_batches + 1):
        end_idx = batch_num * BATCH_SIZE
        df_cumulative = df_full.iloc[:end_idx].copy()
        
        dynamic_metrics = run_dynamic_cumulative(df_cumulative, batch_num, num_batches)
        
        plot_comparison(greedy_metrics, dynamic_metrics, batch_num, num_batches)
        
        print("-" * 70 + "\n")
    
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nGreedy Algorithm (Full {TOTAL_PATIENTS} patients):")
    print(f"  Patients Assigned: {greedy_metrics['patients_assigned']}")
    print(f"  Patients Waiting: {greedy_metrics['patients_waiting']}")
    print(f"  Avg Wait Time: {greedy_metrics['avg_wait_time']:.2f} min")
    print(f"  Utilization Rate: {greedy_metrics['utilization_rate']:.2f}%")
    
    print(f"\nDynamic Algorithm (Final cumulative - all {TOTAL_PATIENTS} patients):")
    print(f"  Patients Assigned: {dynamic_metrics['patients_assigned']}")
    print(f"  Patients Waiting: {dynamic_metrics['patients_waiting']}")
    print(f"  Avg Wait Time: {dynamic_metrics['avg_wait_time']:.2f} min")
    print(f"  Utilization Rate: {dynamic_metrics['utilization_rate']:.2f}%")
    print(f"  Total Urgency Served: {dynamic_metrics['total_urgency_served']:.2f}")
    
    print(f"\n Greedy chart + {num_batches} comparison charts saved!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()