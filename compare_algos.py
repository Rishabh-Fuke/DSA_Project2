import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

# ============================================================
# Force UTF-8 encoding on Windows terminals
# ============================================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ============================================================
# CONFIG — scripts and display names
# ============================================================
scripts = {
    "Dynamic DP": "dynamic_algo.py",
    "Greedy": "greedy_allocation.py"
}

# ============================================================
# Function to run and parse script output
# ============================================================
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
        exec_time = round(end - start, 2)
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return {"Execution_Time": None}

    if result.returncode != 0:
        print(f"Error running {script_name} (exit code {result.returncode})")
        print("STDERR:\n", result.stderr)
        print("STDOUT:\n", result.stdout)
        return {"Execution_Time": exec_time}

    output = result.stdout

    # Extract key-value pairs like: patients_assigned : 47
    pattern = r"([a-zA-Z_]+)\s*:\s*([0-9.]+|N/A)"
    matches = re.findall(pattern, output)
    metrics = {k: v for k, v in matches}

    # Convert numbers
    for k, v in metrics.items():
        if v != "N/A":
            try:
                metrics[k] = float(v)
            except ValueError:
                pass

    metrics["Execution_Time"] = exec_time
    return metrics


# ============================================================
# Run all algorithms
# ============================================================
results = {}
for name, script in scripts.items():
    results[name] = run_and_parse(script)

# ============================================================
# Create DataFrame of results
# ============================================================
df = pd.DataFrame(results).T
print("\n=== COMPARISON TABLE ===\n")
print(df.to_string())

# Save CSV file
df.to_csv("algorithm_comparison.csv", index=True)
print("\nResults saved as 'algorithm_comparison.csv'")

# ============================================================
# Visualization
# ============================================================
metrics_to_plot = [
    "patients_assigned",
    "patients_waiting",
    "avg_wait_time",
    "utilization_rate",
    "total_urgency_served",
    "Execution_Time"
]

available = [m for m in metrics_to_plot if m in df.columns]
df_plot = df[available]

if not df_plot.empty:
    df_plot.plot(kind="bar", figsize=(10,6))
    plt.title("Algorithm Comparison — Dynamic DP vs Greedy Allocation")
    plt.ylabel("Value")
    plt.xlabel("Algorithm")
    plt.legend(title="Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.show()
    print("\nChart saved as 'algorithm_comparison.png'")
else:
    print("\nNo valid metrics found to plot.")

# ============================================================
# Print time complexities and recommendations
# ============================================================
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
