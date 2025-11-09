# Hospital Resource Optimizer

This repository implements two approaches for hospital resource allocation (Doctors and ICU beds): a Time-Indexed Dynamic Programming allocator and a Greedy allocator. The project is designed to evaluate trade-offs between solution quality (minimizing wait and prioritizing urgent patients) and runtime performance.

## Quick summary

- Algorithms included:
	- `dynamic_algo.py` — Time-indexed DP that considers urgency and wait-time when scheduling patients into time slots.
	- `greedy_allocation.py` — Fast greedy allocator that assigns patients to the next available resource using a priority score.
- Utilities:
	- `data_generator.py` — Generate synthetic patient CSV data (small and large samples).
	- `compare_algos.py` — Run both algorithms, collect metrics, save `algorithm_comparison.csv`, and plot a comparison chart.
	- `test_allocation.py` — Unit tests (pytest) for allocator components.

## Repo layout

Files you will care about:

- `data_generator.py` — create `patient_data_short.csv` (by default) for quick experiments.
- `patient_data_short.csv` / `patient_data.csv` — example datasets (short and large). The main scripts expect `patient_data.csv` by default.
- `dynamic_algo.py` — the DP implementation; prints metrics like `patients_assigned`, `avg_wait_time`, `utilization_rate`, etc.
- `greedy_allocation.py` — the greedy implementation; prints similar metrics.
- `compare_algos.py` — runs both algorithms, saves `algorithm_comparison.csv` and `algorithm_comparison.png`.
- `test_allocation.py` — tests for allocation and utility functions (run with `pytest`).
- `requirements.txt` — Python dependencies.

## Requirements

Install dependencies (recommended inside a virtual environment):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

The project depends on:

- pandas
- numpy
- matplotlib
- pytest (for running tests)

Exact versions are listed in `requirements.txt`.

## How to run

1) Generate small sample data (quick):

```powershell
python data_generator.py
```

This script writes `patient_data_short.csv` by default (the script's print message currently mentions `patient_data.csv` — note that mismatch if you see it).

2) Run the Greedy allocator (reads `patient_data.csv` by default):

```powershell
python greedy_allocation.py
```

3) Run the Dynamic (DP) allocator:

```powershell
python dynamic_algo.py
```

Both scripts print metrics to stdout (keys like `patients_assigned`, `avg_wait_time`, `utilization_rate`, `total_urgency_served`) that `compare_algos.py` parses automatically.

4) Compare algorithms and plot results (runs both scripts and saves outputs):

```powershell
python compare_algos.py
```

This will produce:

- `algorithm_comparison.csv` — CSV summary of metrics per algorithm.
- `algorithm_comparison.png` — bar chart comparing metrics.

## Tests

Run unit tests with pytest:

```powershell
pytest -q
```

Note: Some tests in `test_allocation.py` reference helper functions and fields that may be named slightly differently depending on local edits. If a test fails, check variable names in the allocator implementation.

## Input CSV format

The allocator scripts expect a CSV with these columns:

- `patient_id` (int)
- `urgency_score` (float) — higher is more urgent (1–10 scale)
- `arrival_time` (ISO timestamp)
- `treatment_duration` (int, minutes)
- `resource_type` (string) — `Doctor` or `ICU`

Example header:

```
patient_id,urgency_score,arrival_time,treatment_duration,resource_type
```

## Notes and small known issues

- `data_generator.py` currently writes `patient_data_short.csv` but prints a message mentioning `patient_data.csv`. This is a minor messaging bug and does not affect file output.
- `visualize.py` is referenced in an older README but is not present in this repository; use `compare_algos.py` which generates comparison charts from algorithm outputs.

## Suggested next steps / improvements

- Add CLI argument parsing to `greedy_allocation.py` and `dynamic_algo.py` so the input file path and resource counts can be passed at runtime (currently hard-coded values are used in the main scripts).
- Add a small `runner.py` to allow running with different dataset sizes, seeds, and resource configurations easily.
- Harmonize output metric keys across both allocators so `compare_algos.py` does not need to handle `N/A` or missing columns.

## Contact

If you have questions about the code, tests, or data, open an issue or contact the repository authors.