import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_distribution(labels, work_dir, mode):
    """Visualize class distribution."""
    try:
        values, count = np.unique(labels, return_counts=True)
        plt.figure()
        plt.bar(values, count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.savefig(f'{work_dir}/{mode}_Label_Distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing distribution: {e}")

def plot_loss_curves(train_loss, val_loss, work_dir, subject_id):
    """Visualize training and validation loss curves."""
    try:
        if not train_loss or not val_loss:
            return
            
        epochs = range(len(train_loss))
        
        plt.figure()
        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title(f'Train vs Val Loss for {subject_id}')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'{work_dir}/trainvsval_{subject_id}.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing loss: {e}")

def plot_confusion_matrix(y_pred, y_true, work_dir):
    """Visualize confusion matrix."""
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(np.unique(y_true))
        plt.yticks(np.unique(y_true))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.savefig(f'{work_dir}/Confusion_Matrix.png')
        plt.close()
    except Exception as e:
        print(f"Error visualizing confusion matrix: {e}")
