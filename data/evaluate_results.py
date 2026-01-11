import json

def calculate_metrics(data):
    """Calculate total counts for correctness, hallucination, and source drift."""
    total_correct = 0
    total_hallucination = 0
    total_source_drift = 0
    
    for item in data:
        total_correct += item.get('correctness', 0)
        total_hallucination += item.get('hallucination', 0)
        total_source_drift += item.get('source_drift', 0)
    
    return {
        'total_correctness': total_correct,
        'total_hallucination': total_hallucination,
        'total_source_drift': total_source_drift,
        'total_questions': len(data)
    }

def calculate_percentages(metrics):
    """Calculate percentages for each metric."""
    total = metrics['total_questions']
    
    return {
        'correctness_percentage': (metrics['total_correctness'] / total) * 100 if total > 0 else 0,
        'hallucination_percentage': (metrics['total_hallucination'] / total) * 100 if total > 0 else 0,
        'source_drift_percentage': (metrics['total_source_drift'] / total) * 100 if total > 0 else 0,
    }


def load_and_analyze(filename):
    """Load data from JSON file and analyze it."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = calculate_metrics(data)
    percentages = calculate_percentages(metrics)
    
    print("=== EVALUATION RESULTS ===")
    print(f"Total Questions: {metrics['total_questions']}")
    print()
    print("Absolute Counts:")
    print(f"  Correct Answers: {metrics['total_correctness']}")
    print(f"  Hallucinations: {metrics['total_hallucination']}")
    print(f"  Source Drift: {metrics['total_source_drift']}")
    print()
    print("Percentages:")
    print(f"  Correctness: {percentages['correctness_percentage']:.2f}%")
    print(f"  Hallucination: {percentages['hallucination_percentage']:.2f}%")
    print(f"  Source Drift: {percentages['source_drift_percentage']:.2f}%")
    
    return metrics, percentages


metrics, percentages = load_and_analyze('/app/data/judged_clean_rag_k3.json')

# run: docker compose exec app python data/evaluate_results.py

