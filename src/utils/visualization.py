
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_distribution(labels, work_dir, mode, logger=None):
    """Visualize class distribution"""
    try:
        values, count = np.unique(labels, return_counts=True)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(values, count, color=['blue', 'red'])
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.xticks(values)

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom')

        # Save visualization
        viz_dir = os.path.join(work_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, f'{mode}_distribution.png'))
        plt.close()

        # Log distribution
        if logger:
            dist_str = ", ".join([f"Label {int(v)}: {c}" for v, c in zip(values, count)])
            logger(f"{mode} distribution: {dist_str}")

    except Exception as e:
        if logger:
            logger(f"Error visualizing distribution: {e}")

def plot_loss_curves(train_loss, val_loss, work_dir, subject_id, logger=None):
    """Visualize training and validation loss curves"""
    try:
        if not train_loss or not val_loss:
            if logger:
                logger("No loss data to visualize")
            return

        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title(f'Training vs Validation Loss (Subject {subject_id})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Save visualization
        viz_dir = os.path.join(work_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, f'loss_curves_{subject_id}.png'))
        plt.close()

    except Exception as e:
        if logger:
            logger(f"Error visualizing loss: {e}")

def plot_confusion_matrix(y_pred, y_true, work_dir, subject_id, logger=None):
    """Visualize confusion matrix"""
    try:
        # Convert to numpy arrays
        if isinstance(y_pred, (list, tuple)):
            y_pred = np.array(y_pred)
        if isinstance(y_true, (list, tuple)):
            y_true = np.array(y_true)

        # Ensure arrays are flattened
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix (Subject {subject_id})')

        # Set labels based on unique values
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        if len(classes) == 2:
            class_labels = ['ADL', 'Fall']
        else:
            class_labels = [str(c) for c in classes]

        plt.xticks(np.arange(len(classes)) + 0.5, class_labels)
        plt.yticks(np.arange(len(classes)) + 0.5, class_labels)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save visualization
        viz_dir = os.path.join(work_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{subject_id}.png'))
        plt.close()

    except Exception as e:
        if logger:
            logger(f"Error visualizing confusion matrix: {e}")
