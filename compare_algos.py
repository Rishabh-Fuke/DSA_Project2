import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys


try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


scripts = {
    "Dynamic DP": "dynamic_algo.py",
    "Greedy": "greedy_allocation.py"
}


def run_and_parse(script_name):
    print(f"\nRunning {script_name}...\n")
    start = time.time()

    try:
        result = subprocess.run(
            ["python", script_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        end = time.time()
        exec_time_sec = end - start
        exec_time_ms = round(exec_time_sec * 10, 2)
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return {"Execution_Time_*10": None}

    if result.returncode != 0:
        print(f"Error running {script_name} (exit code {result.returncode})")
        print("STDERR:\n", result.stderr)
        print("STDOUT:\n", result.stdout)
        return {"Execution_Time_*10": exec_time_ms}

    output = result.stdout

    pattern = r"([a-zA-Z_]+)\s*:\s*([0-9.]+|N/A)"
    matches = re.findall(pattern, output)
    metrics = {k: v for k, v in matches}

    for k, v in metrics.items():
        if v != "N/A":
            try:
                metrics[k] = float(v)
            except ValueError:
                pass

    metrics["Execution_Time_*10"] = exec_time_ms
    return metrics


# Run all scripts
results = {}
for name, script in scripts.items():
    results[name] = run_and_parse(script)

# Create DataFrame
df = pd.DataFrame(results).T
print("\n=== COMPARISON TABLE ===\n")
print(df.to_string())

df.to_csv("algorithm_comparison.csv", index=True)
print("\nResults saved as 'algorithm_comparison.csv'")

# Metrics to plot
metrics_to_plot = [
    "patients_assigned",
    "patients_waiting",
    "avg_wait_time",
    "utilization_rate",
    "total_urgency_served",
    "Execution_Time_*10"
]

available = [m for m in metrics_to_plot if m in df.columns]
df_plot = df[available]

# Plot chart
# Plot chart
if not df_plot.empty:
    ax = df_plot.plot(kind="bar", figsize=(10,6))
    plt.title("Algorithm Comparison — Dynamic DP vs Greedy Allocation")
    plt.ylabel("Value")
    plt.xlabel("Algorithm")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Make Execution_Time_*10 bars translucent
    for i, bar in enumerate(ax.patches):
        metric_idx = i % len(df_plot.columns)  # column index
        col_name = df_plot.columns[metric_idx]
        if col_name == "Execution_Time_*10":
            bar.set_alpha(0.4)  # translucent
        else:
            bar.set_alpha(1.0)  # solid

        # Annotate all bars
        height = bar.get_height()
        ax.annotate(f'{height}', 
                    (bar.get_x() + bar.get_width()/2, height),
                    ha='center', va='bottom', fontsize=8, rotation=90)

    plt.legend(title="Metrics")
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.show()
    print("\nChart saved as 'algorithm_comparison.png'")
else:
    print("\nNo valid metrics found to plot.")


# Time complexities and insights
print("\n=== TIME COMPLEXITIES ===")
print("Dynamic DP : O(n × t) — depends on number of patients × time slots")
print("Greedy     : O(n log n) — sorting + allocation decisions")

print("\n=== PERFORMANCE INSIGHTS ===")
print("""
Dynamic Programming (DP):
- Optimized scheduling considering urgency and time slots.
- Reduces average and total waiting time.
- Balances load across doctors and ICUs.
- Ideal for emergency or critical care settings.

Greedy Allocation:
- Fast and simple, works well for large patient volumes.
- May increase waiting time but uses resources efficiently.
- Ideal for outpatient or routine hospital scheduling.
""")

print("\n=== RECOMMENDATIONS ===")
print("""
Emergency or ICU Scheduling -> Use Dynamic DP (accuracy over speed)
Large-scale outpatient flow  -> Use Greedy (speed over accuracy)
Limited doctors/resources    -> Dynamic DP performs better
Many short appointments      -> Greedy scales efficiently
""")

print("\nComparison complete. Results and chart generated successfully.")
