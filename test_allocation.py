import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from greedy_allocation import GreedyAllocator, ResourcePool

@pytest.fixture
def sample_data():
    """Create a small synthetic dataset for testing."""
    data = {
        'patient_id': range(1, 11),
        'urgency_score': [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 10.0],
        'arrival_time': [
            datetime(2025, 1, 1, 0, i) for i in range(10)
        ],
        'treatment_duration': [60, 30, 45, 90, 120, 60, 30, 45, 90, 120],
        'resource_type': ['Doctor', 'ICU', 'Doctor', 'ICU', 'Doctor', 
                         'ICU', 'Doctor', 'ICU', 'Doctor', 'ICU']
    }
    return pd.DataFrame(data)

def test_resource_pool_initialization():
    """Test ResourcePool initialization and resource tracking."""
    pool = ResourcePool(num_doctors=3, num_icu=2)
    assert len(pool.doctors) == 3
    assert len(pool.icu_beds) == 2
    assert len(pool.doctor_assignments) == 0
    assert len(pool.icu_assignments) == 0

def test_resource_assignment():
    """Test resource assignment and freeing."""
    pool = ResourcePool(num_doctors=1, num_icu=1)
    start_time = datetime(2025, 1, 1, 0, 0)
    
    # Assign doctor
    pool.assign_resource("Doctor", 0, start_time, 60)
    assert 0 in pool.doctor_assignments
    
    # Check resource is busy
    resource_id, next_time = pool.get_next_available_resource("Doctor", start_time)
    assert next_time > start_time
    
    # Check resource frees after time passes
    resource_id, next_time = pool.get_next_available_resource(
        "Doctor", 
        start_time + timedelta(minutes=61)
    )
    assert resource_id == 0
    assert next_time == start_time + timedelta(minutes=61)

def test_priority_queue_ordering(sample_data):
    """Test that patients are processed in urgency order."""
    allocator = GreedyAllocator(num_doctors=1, num_icu=1)
    allocator.allocate_resources(sample_data)
    
    # Check doctor allocations are in urgency order
    doctor_allocs = [
        a for a in allocator.resource_utilization["Doctor"]
        if a['urgency_score'] > 0
    ]
    urgency_scores = [a['urgency_score'] for a in doctor_allocs]
    assert urgency_scores == sorted(urgency_scores, reverse=True)

def test_capacity_analysis(sample_data):
    """Test capacity analysis calculations."""
    analysis = GreedyAllocator.analyze_capacity_requirements(sample_data)
    
    assert 'Doctor' in analysis
    assert 'ICU' in analysis
    for resource in ['Doctor', 'ICU']:
        assert analysis[resource]['min_resources_needed'] > 0
        assert analysis[resource]['recommended_resources'] >= analysis[resource]['min_resources_needed']
        assert analysis[resource]['arrival_rate_per_hour'] > 0
        assert analysis[resource]['avg_treatment_mins'] > 0

def test_no_resources():
    """Test behavior when no resources are available."""
    data = {
        'patient_id': [1],
        'urgency_score': [9.0],
        'arrival_time': [datetime(2025, 1, 1, 0, 0)],
        'treatment_duration': [60],
        'resource_type': ['Doctor']
    }
    df = pd.DataFrame(data)
    
    allocator = GreedyAllocator(num_doctors=0, num_icu=0)
    allocator.allocate_resources(df)
    
    metrics = allocator.get_metrics()
    assert metrics['Doctor_patients_processed'] == 0
    assert metrics['Doctor_unassigned'] > 0

def test_all_same_resource():
    """Test when all patients need the same resource."""
    data = {
        'patient_id': range(1, 6),
        'urgency_score': [9.0, 8.0, 7.0, 6.0, 5.0],
        'arrival_time': [
            datetime(2025, 1, 1, 0, i) for i in range(5)
        ],
        'treatment_duration': [60] * 5,
        'resource_type': ['Doctor'] * 5
    }
    df = pd.DataFrame(data)
    
    allocator = GreedyAllocator(num_doctors=1, num_icu=1)
    allocator.allocate_resources(df)
    
    metrics = allocator.get_metrics()
    assert metrics['Doctor_patients_processed'] == 5
    assert metrics['ICU_patients_processed'] == 0
    assert metrics['Doctor_utilization'] > 0
    assert metrics['ICU_utilization'] == 0

def test_zero_duration():
    """Test handling of zero duration treatments."""
    data = {
        'patient_id': [1],
        'urgency_score': [9.0],
        'arrival_time': [datetime(2025, 1, 1, 0, 0)],
        'treatment_duration': [0],
        'resource_type': ['Doctor']
    }
    df = pd.DataFrame(data)
    
    allocator = GreedyAllocator(num_doctors=1, num_icu=1)
    allocator.allocate_resources(df)
    
    metrics = allocator.get_metrics()
    assert metrics['Doctor_patients_processed'] == 1
    assert metrics['Doctor_utilization'] == 0

def test_metrics_calculation(sample_data):
    """Test that metrics are calculated correctly."""
    allocator = GreedyAllocator(num_doctors=2, num_icu=2)
    allocator.allocate_resources(sample_data)
    
    metrics = allocator.get_metrics()
    
    # Check all required metrics are present
    required_metrics = [
        'average_wait_time', 'max_wait_time', 'min_wait_time',
        'total_patients_processed',
        'Doctor_utilization', 'ICU_utilization',
        'Doctor_patients_processed', 'ICU_patients_processed'
    ]
    for metric in required_metrics:
        assert metric in metrics
        
    # Check metric values are reasonable
    assert metrics['total_patients_processed'] <= len(sample_data)
    assert metrics['Doctor_utilization'] <= 100
    assert metrics['ICU_utilization'] <= 100
    assert metrics['average_wait_time'] >= 0