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
        exec_time_ms = round(exec_time_sec * 1000, 2)  # convert to milliseconds
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return {"Execution_Time_ms": None}

    if result.returncode != 0:
        print(f"Error running {script_name} (exit code {result.returncode})")
        print("STDERR:\n", result.stderr)
        print("STDOUT:\n", result.stdout)
        return {"Execution_Time_ms": exec_time_ms}

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

    metrics["Execution_Time_ms"] = exec_time_ms
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
    "Execution_Time_ms"
]

available = [m for m in metrics_to_plot if m in df.columns]
df_plot = df[available]

# Plot chart
if not df_plot.empty:
    ax = df_plot.plot(kind="bar", figsize=(10,6))
    plt.title("Algorithm Comparison — Dynamic DP vs Greedy Allocation")
    plt.ylabel("Value")
    plt.xlabel("Algorithm")
    plt.legend(title="Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate execution time bars at the bottom
    for p in ax.patches:
        height = p.get_height()
        if p.get_x() % 1 == 0:  # rough check for bar
            ax.annotate(f'{height}', (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=8, rotation=90)

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
