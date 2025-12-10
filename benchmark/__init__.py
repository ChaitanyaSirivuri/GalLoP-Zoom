"""
MuSciClaims Benchmark Package

This package contains tools for benchmarking vision-language models
on the MuSciClaims scientific claim verification dataset.
"""

from .data_loader import load_dataset_with_images, get_image_for_sample
from .prompts import get_prompt_d, get_prompt_rd, parse_decision, normalize_label
from .metrics import calculate_metrics, print_metrics, save_metrics_to_csv, save_predictions_to_csv

__all__ = [
    'load_dataset_with_images',
    'get_image_for_sample',
    'get_prompt_d',
    'get_prompt_rd',
    'parse_decision',
    'normalize_label',
    'calculate_metrics',
    'print_metrics',
    'save_metrics_to_csv',
    'save_predictions_to_csv',
]
