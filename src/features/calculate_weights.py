import pandas as pd
import numpy as np
import json
from pathlib import Path

# Define paths using absolute paths based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# The 15 expression labels based on the EEV paper (lowercase to match CSV)
EXPRESSION_COLUMNS = [
    'amusement', 'anger', 'awe', 'concentration', 'confusion',
    'contempt', 'contentment', 'disappointment', 'doubt', 
    'elation', 'interest', 'pain', 'sadness', 'surprise', 'triumph'
]

def calculate_class_weights():
    """Calculates inverse frequency class weights to balance dataset decay."""
    train_csv = PROCESSED_DIR / "train_pruned.csv"
    
    if not train_csv.exists():
        print("Pruned dataset not found.")
        return

    print("Calculating class distribution weights...")
    df = pd.read_csv(train_csv)
    
    # Sum the confidence scores across all rows for each expression
    class_sums = df[EXPRESSION_COLUMNS].sum().to_dict()
    
    # Compute inverse frequency weights
    # Formula: total_weight / (num_classes * class_sum)
    total_samples = df[EXPRESSION_COLUMNS].sum().sum()
    num_classes = len(EXPRESSION_COLUMNS)
    
    class_weights = {}
    for expression, score_sum in class_sums.items():
        if score_sum > 0:
            weight = total_samples / (num_classes * score_sum)
        else:
            weight = 0.0 # Handle edge case if a class is entirely deleted
        class_weights[expression] = round(weight, 4)
        
    # Export to JSON
    output_file = WEIGHTS_DIR / "class_weights.json"
    with open(output_file, 'w') as f:
        json.dump(class_weights, f, indent=4)
        
    print(f"Class weights successfully saved to {output_file}")
    for exp, w in class_weights.items():
        print(f" - {exp}: {w}")

if __name__ == "__main__":
    calculate_class_weights()