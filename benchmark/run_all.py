"""
Run all benchmarks and generate comparison report.
"""

import os
import sys
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_benchmarks():
    """Run all model benchmarks and generate comparison."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    all_metrics = {}
    
    # Run Qwen2.5-VL benchmark
    print("\n" + "="*60)
    print("Running Qwen2.5-VL-7B-Instruct Benchmark")
    print("="*60)
    try:
        from benchmark.benchmark_qwen25_vl import run_benchmark as run_qwen25
        metrics, _ = run_qwen25()
        all_metrics["Qwen2.5-VL-7B"] = metrics
    except Exception as e:
        print(f"Error running Qwen2.5-VL benchmark: {e}")
    
    # Run Qwen3-VL benchmark
    print("\n" + "="*60)
    print("Running Qwen3-VL-30B-A3B-Instruct Benchmark")
    print("="*60)
    try:
        from benchmark.benchmark_qwen3_vl import run_benchmark as run_qwen3
        metrics, _ = run_qwen3()
        all_metrics["Qwen3-VL-30B"] = metrics
    except Exception as e:
        print(f"Error running Qwen3-VL benchmark: {e}")
    
    # Run LLaVA-Next benchmark
    print("\n" + "="*60)
    print("Running LLaVA-Next Benchmark")
    print("="*60)
    try:
        from benchmark.benchmark_llava_next import run_benchmark as run_llava
        metrics, _ = run_llava()
        all_metrics["LLaVA-Next-7B"] = metrics
    except Exception as e:
        print(f"Error running LLaVA-Next benchmark: {e}")
    
    # Generate comparison report
    if all_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(output_dir, f"comparison_report_{timestamp}.csv")
        
        with open(comparison_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Model", 
                "SUPPORT_P", "SUPPORT_R", "SUPPORT_F1",
                "NEUTRAL_P", "NEUTRAL_R", "NEUTRAL_F1",
                "CONTRADICT_P", "CONTRADICT_R", "CONTRADICT_F1",
                "OVERALL_P", "OVERALL_R", "OVERALL_F1",
                "Accuracy"
            ])
            
            for model_name, metrics in all_metrics.items():
                row = [model_name]
                for cls in ["SUPPORT", "NEUTRAL", "CONTRADICT", "OVERALL"]:
                    m = metrics[cls]
                    row.extend([
                        f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}",
                        f"{m['f1']:.4f}",
                    ])
                row.append(f"{metrics['OVERALL']['accuracy']:.4f}")
                writer.writerow(row)
        
        print(f"\n" + "="*60)
        print("COMPARISON REPORT")
        print("="*60)
        print(f"\nResults saved to: {comparison_path}")
        
        # Print summary
        print("\n{:<20} {:<12} {:<12} {:<12} {:<12}".format(
            "Model", "Overall P", "Overall R", "Overall F1", "Accuracy"
        ))
        print("-" * 68)
        for model_name, metrics in all_metrics.items():
            m = metrics["OVERALL"]
            print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                model_name, m['precision'], m['recall'], m['f1'], m['accuracy']
            ))


if __name__ == "__main__":
    run_all_benchmarks()
