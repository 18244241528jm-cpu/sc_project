import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from pathlib import Path

def plot_confusion_matrix(y_true, y_pred, save_path: Path, title: str = "Confusion Matrix"):
    """
    Plot and save the confusion matrix.
    """
    # HAR Dataset class names
    class_names = [
        "Walking", 
        "Walking Up", 
        "Walking Down", 
        "Sitting", 
        "Standing", 
        "Laying"
    ]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,         # Show numbers in cells
        fmt='d',            # Integer format
        cmap='Blues',       # Color map
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_model_comparison(results: dict, save_path: Path):
    """
    Plot accuracy comparison for different models.
    
    Parameters:
    ----------
    results : dict
        Dictionary mapping model names to accuracy scores.
        Example: {"Logistic Regression": 0.95, "SVM": 0.96}
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    models = list(results.keys())
    scores = list(results.values())
    
    # Plot bar chart
    bars = plt.bar(models, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Set Y-axis limit for better visibility of differences
    plt.ylim(0.8, 1.0)
    plt.title("Model Comparison (Accuracy)")
    plt.ylabel("Accuracy Score")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.2%}',
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")
