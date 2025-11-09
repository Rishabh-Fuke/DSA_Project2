# Hospital Resource Optimizer

A hospital resource scheduling system that compares Dynamic Programming and Greedy approaches for allocating doctors and ICU beds to patients.

## Quick Start

### Prerequisites
- Python 3.6+
- Virtual environment (recommended)

### Setup and Run

```powershell
# Clone and setup environment
git clone https://github.com/Rishabh-Fuke/DSA_Project2.git
cd DSA_Project2
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt

# For complete comparison analysis
python analysis.py

# Or run components individually:

# Generate dataset
python data_generator.py

# Run Dynamic Programming algorithm only
python dynamic_algo.py

# Run Greedy algorithm only
python greedy_allocation.py
```

## What the Analysis Does

Running `analysis.py` will:
1. Create or load patient dataset (default: 100,000 patients)
2. Run both allocation algorithms:
   - Greedy Algorithm (fast, real-time allocation)
   - Dynamic Programming (optimal, time-indexed scheduling)
3. Generate performance comparisons:
   - `algorithm_comparison.csv`: Detailed metrics
   - `algorithm_comparison.png`: Visual comparison
   - `comparison_batch_*.png`: Batch analysis charts

## Project Structure

```
DSA_Project2/
├── analysis.py              # Main script for analysis and visualization
├── dynamic_algo.py         # DP algorithm implementation
├── greedy_allocation.py    # Greedy algorithm implementation
├── data_generator.py       # Data generation utility
├── requirements.txt        # Project dependencies
├── patient_data.csv        # Generated dataset
└── comparison_batch_*.png  # Analysis visualizations


## Output Metrics

The analysis generates comprehensive performance metrics:
- Number of patients assigned
- Average wait time
- Resource utilization rate
- Total urgency score served
- Processing time per algorithm

## Running Individual Components

You can run each part of the system separately:

1. **Generate Data** (`data_generator.py`):
   - Creates synthetic patient dataset
   - Outputs to `patient_data.csv`
   - Configurable number of patients

2. **Dynamic Programming** (`dynamic_algo.py`):
   - Runs the DP algorithm only
   - Optimal time-indexed scheduling
   - Shows detailed metrics for DP approach

3. **Greedy Algorithm** (`greedy_allocation.py`):
   - Runs the Greedy algorithm only
   - Fast real-time allocation
   - Shows detailed metrics for Greedy approach

4. **Full Comparison** (`analysis.py`):
   - Runs both algorithms
   - Generates comparative visualizations
   - Produces detailed performance metrics