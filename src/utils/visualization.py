#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module for LightHART-TF

Contains functions for visualizing data and results
"""
import os
import numpy as np
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
        else:
            print(f"Error visualizing distribution: {e}")

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
        else:
            print(f"Error visualizing loss: {e}")

def plot_confusion_matrix(y_pred, y_true, work_dir, subject_id, logger=None):
    """Visualize confusion matrix"""
    try:
        # Convert to numpy arrays
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Subject {subject_id})')
        
        # Set labels
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks + 0.5, classes)
        plt.yticks(tick_marks + 0.5, classes)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save visualization
        viz_dir = os.path.join(work_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{subject_id}.png'))
        plt.close()
        
        # Log confusion matrix
        if logger:
            logger(f"Confusion matrix for subject {subject_id}:")
            for i in range(cm.shape[0]):
                row_str = " ".join([f"{cm[i, j]:4d}" for j in range(cm.shape[1])])
                logger(f"  {row_str}")
        
    except Exception as e:
        if logger:
            logger(f"Error visualizing confusion matrix: {e}")
        else:
            print(f"Error visualizing confusion matrix: {e}")
