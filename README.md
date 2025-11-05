# Hospital Resource Optimizer

A Python-based hospital resource optimization system that uses Greedy and Dynamic Programming approaches to efficiently allocate doctors and ICU beds to patients, minimizing wait times while maintaining fairness.

## Features

- **Smart Resource Allocation**: Prioritizes patients based on urgency scores and waiting time
- **Capacity Analysis**: Uses Little's Law to recommend optimal resource levels
- **Multiple Resource Types**: Handles both doctors and ICU bed allocations
- **Performance Metrics**: Tracks wait times, resource utilization, and fairness
- **Visualization**: Provides insights through heatmaps and timeline plots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Rishabh-Fuke/DSA_Project2.git
cd DSA_Project2
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Sample Data

Generate a dataset with 100,000 patient records:
```bash
python data_generator.py
```

### Run Resource Allocation

Basic usage with default settings:
```bash
python greedy_allocation.py
```

Specify resource counts and sample size:
```bash
python greedy_allocation.py --doctors 20 --icu 10 --sample 1000
```

### Generate Visualizations

Create resource analysis plots:
```bash
python visualize.py --sample 1000 --hours 24
```

This will generate:
- `resource_analysis.png`: Heatmaps showing wait times and utilization
- `resource_timeline.png`: Timeline of resource usage

### Run Tests

Run the test suite:
```bash
pytest test_allocation.py -v
```

## Project Structure

- `data_generator.py`: Generates synthetic patient data
- `greedy_allocation.py`: Implements greedy resource allocation algorithm
- `visualize.py`: Creates analysis plots and visualizations
- `test_allocation.py`: Unit tests for allocation logic
- `requirements.txt`: Project dependencies

## Input Data Format

The system expects a CSV file with the following columns:
- `patient_id` (int): Unique identifier
- `urgency_score` (float): 1-10 scale, higher is more urgent
- `arrival_time` (timestamp): When the patient arrives
- `treatment_duration` (int): Minutes needed for treatment
- `resource_type` (string): "ICU" or "Doctor"

## Performance Metrics

The system tracks multiple metrics:
- Average and maximum wait times (overall and per resource)
- Resource utilization percentages
- Number of patients processed/unassigned
- Recommended resource levels based on arrival patterns

## Capacity Analysis

Uses Little's Law (L = λW) to calculate minimum required resources:
- L: Average number of patients in system
- λ: Arrival rate (patients/hour)
- W: Average treatment duration

Adds buffer margins for:
- Peak time surges (+20%)
- Emergency capacity (+50% for worst case)

## Team

- Rishabh Fuke: Data preparation, Dynamic Programming implementation
- Neerav Gandhi: Greedy algorithm implementation
- Siddhant Pallod: Visualization, testing, performance comparison

## License

MIT License