"""
Metrics calculation for MuSciClaims benchmark.
Computes Precision, Recall, F1 for each class and overall.
"""

import csv
from typing import List, Dict
from collections import Counter


def calculate_metrics(predictions: List[str], labels: List[str]) -> Dict:
    """
    Calculate precision, recall, and F1 score for each class and overall.
    
    Args:
        predictions: List of predicted labels
        labels: List of ground truth labels
        
    Returns:
        Dictionary with metrics for each class and overall
    """
    classes = ["SUPPORT", "NEUTRAL", "CONTRADICT"]
    
    # Count true positives, false positives, false negatives for each class
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    
    for pred, label in zip(predictions, labels):
        for c in classes:
            if pred == c and label == c:
                tp[c] += 1
            elif pred == c and label != c:
                fp[c] += 1
            elif pred != c and label == c:
                fn[c] += 1
    
    # Calculate per-class metrics
    metrics = {}
    
    for c in classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp[c],
            "fp": fp[c],
            "fn": fn[c],
        }
    
    # Calculate overall (macro-averaged) metrics
    overall_precision = sum(metrics[c]["precision"] for c in classes) / len(classes)
    overall_recall = sum(metrics[c]["recall"] for c in classes) / len(classes)
    overall_f1 = sum(metrics[c]["f1"] for c in classes) / len(classes)
    
    # Calculate accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if len(labels) > 0 else 0.0
    
    metrics["OVERALL"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "accuracy": accuracy,
        "total": len(labels),
        "correct": correct,
    }
    
    # Count predictions and labels distribution
    pred_counts = Counter(predictions)
    label_counts = Counter(labels)
    
    metrics["distribution"] = {
        "predictions": dict(pred_counts),
        "labels": dict(label_counts),
    }
    
    return metrics


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    
    print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    
    for cls in ["SUPPORT", "NEUTRAL", "CONTRADICT"]:
        m = metrics[cls]
        print(f"{cls:<12} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}")
    
    print("-" * 48)
    m = metrics["OVERALL"]
    print(f"{'OVERALL':<12} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}")
    print(f"\nAccuracy: {m['accuracy']:.4f} ({m['correct']}/{m['total']})")
    
    # Print distribution
    print(f"\nPrediction Distribution: {metrics['distribution']['predictions']}")
    print(f"Label Distribution: {metrics['distribution']['labels']}")


def save_metrics_to_csv(metrics: Dict, output_path: str, model_name: str = "Model"):
    """
    Save metrics to a CSV file.
    
    Format:
    Model, Class, Precision, Recall, F1
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Model", "Class", "Precision", "Recall", "F1"])
        
        # Per-class metrics
        for cls in ["SUPPORT", "NEUTRAL", "CONTRADICT", "OVERALL"]:
            m = metrics[cls]
            writer.writerow([
                model_name,
                cls,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
            ])
    
    print(f"Metrics saved to {output_path}")


def save_predictions_to_csv(results: List[Dict], output_path: str):
    """
    Save detailed predictions to a CSV file.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Index", "Claim", "Caption", "Label", "Prediction", 
            "Correct", "Image_Path", "Panels", "Raw_Response"
        ])
        
        for r in results:
            writer.writerow([
                r.get("index", ""),
                r.get("claim", "")[:200],  # Truncate long claims
                r.get("caption", "")[:200],  # Truncate long captions
                r.get("label", ""),
                r.get("prediction", ""),
                "Yes" if r.get("correct", False) else "No",
                r.get("image_path", ""),
                str(r.get("panels", [])),
                r.get("raw_response", "")[:500],  # Truncate long responses
            ])
    
    print(f"Predictions saved to {output_path}")
