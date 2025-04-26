import tensorflow as tf

class EarlyStoppingTF:
    """Early stopping callback for TensorFlow."""
    def __init__(self, patience=15, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class CheckpointManagerTF:
    """Manages TensorFlow checkpoints for model saving/loading."""
    def __init__(self, model, optimizer, model_path, max_to_keep=3):
        self.model = model
        self.optimizer = optimizer
        self.model_path = model_path
        self.ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=model_path, max_to_keep=max_to_keep)
        self.best_loss = float('inf')
    
    def save_checkpoint(self, val_loss, epoch):
        """Save checkpoint if validation loss improves."""
        path = self.manager.save(checkpoint_number=epoch)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            return path
        return None
    
    def load_best_checkpoint(self):
        """Load the best checkpoint."""
        status = self.ckpt.restore(self.manager.latest_checkpoint)
        return status.expect_partial()
