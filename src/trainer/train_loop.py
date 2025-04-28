import time
import tensorflow as tf
import numpy as np
import logging
from utils.metrics import calculate_metrics

class TrainingManager:
    def __init__(self, model, optimizer, criterion, train_data, val_data, test_data=None,
                 max_epochs=100, patience=15, work_dir='./experiments', model_name='model',
                 save_tflite=True, logger=logging.getLogger()):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.max_epochs = max_epochs
        self.patience = patience
        self.work_dir = work_dir
        self.model_name = model_name
        self.save_tflite = save_tflite
        self.logger = logger
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        
        os.makedirs(os.path.join(work_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'results'), exist_ok=True)
    
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            outputs = self.model(inputs, training=True)
            
            if isinstance(outputs, tuple) and len(outputs) > 0:
                logits = outputs[0]
            else:
                logits = outputs
            
            if len(logits.shape) > 1 and logits.shape[-1] > 1:
                loss = self.criterion(targets, logits)
            else:
                loss = self.criterion(targets, tf.squeeze(logits))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            predictions = tf.argmax(logits, axis=-1)
        else:
            predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
        
        return loss, predictions
    
    def eval_step(self, inputs, targets):
        outputs = self.model(inputs, training=False)
        
        if isinstance(outputs, tuple) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = outputs
        
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            loss = self.criterion(targets, logits)
        else:
            loss = self.criterion(targets, tf.squeeze(logits))
        
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            predictions = tf.argmax(logits, axis=-1)
        else:
            predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
        
        return loss, predictions, logits
    
    def train_epoch(self, epoch):
        self.model.trainable = True
        start_time = time.time()
        
        train_loss = 0.0
        all_labels = []
        all_preds = []
        steps = 0
        
        total_batches = len(self.train_data)
        self.logger.info(f"Epoch {epoch+1}/{self.max_epochs}: training on {total_batches} batches")
        
        for batch_idx in range(total_batches):
            try:
                inputs, targets, _ = self.train_data[batch_idx]
                targets = tf.cast(targets, tf.float32)
                loss, predictions = self.train_step(inputs, targets)
                
                train_loss += loss.numpy()
                all_labels.extend(targets.numpy())
                all_preds.extend(predictions.numpy())
                steps += 1
                
                if batch_idx % 10 == 0 or batch_idx + 1 == total_batches:
                    self.logger.info(f"  Batch {batch_idx+1}/{total_batches}, Loss: {loss.numpy():.4f}")
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
        
        if steps == 0:
            self.logger.error("No valid training steps completed in this epoch")
            return float('inf'), None
        
        train_loss /= steps
        metrics = calculate_metrics(all_labels, all_preds)
        
        epoch_time = time.time() - start_time
        self.logger.info(
            f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
            f"Loss: {train_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.2f}%, "
            f"F1: {metrics['f1']:.2f}%, "
            f"Prec: {metrics['precision']:.2f}%, "
            f"Rec: {metrics['recall']:.2f}%, "
            f"AUC: {metrics['auc']:.2f}%"
        )
        
        self.train_loss_history.append(float(train_loss))
        return train_loss, metrics
    
    def evaluate(self, dataset, name="val"):
        self.model.trainable = False
        start_time = time.time()
        
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        all_logits = []
        steps = 0
        
        total_batches = len(dataset)
        self.logger.info(f"Evaluating {name} set: {total_batches} batches")
        
        for batch_idx in range(total_batches):
            try:
                inputs, targets, _ = dataset[batch_idx]
                targets = tf.cast(targets, tf.float32)
                loss, predictions, logits = self.eval_step(inputs, targets)
                
                eval_loss += loss.numpy()
                all_labels.extend(targets.numpy())
                all_preds.extend(predictions.numpy())
                
                if isinstance(logits, tf.Tensor):
                    if len(logits.shape) == 1 or (len(logits.shape) > 1 and logits.shape[-1] == 1):
                        all_logits.extend(tf.squeeze(logits).numpy())
                steps += 1
                
                if batch_idx % 10 == 0 or batch_idx + 1 == total_batches:
                    self.logger.info(f"  Batch {batch_idx+1}/{total_batches}")
            except Exception as e:
                self.logger.error(f"Error in {name} batch {batch_idx}: {e}")
        
        if steps == 0:
            self.logger.error(f"No valid evaluation steps completed for {name}")
            return float('inf'), None
        
        eval_loss /= steps
        metrics = calculate_metrics(all_labels, all_preds)
        
        eval_time = time.time() - start_time
        self.logger.info(
            f"{name.capitalize()} evaluation completed in {eval_time:.2f}s - "
            f"Loss: {eval_loss:.4f}, "
            f"Acc: {metrics['accuracy']:.2f}%, "
            f"F1: {metrics['f1']:.2f}%, "
            f"Prec: {metrics['precision']:.2f}%, "
            f"Rec: {metrics['recall']:.2f}%, "
            f"AUC: {metrics['auc']:.2f}%"
        )
        
        if name == "val":
            self.val_loss_history.append(float(eval_loss))
        
        return eval_loss, metrics
    
    def save_model(self, epoch, subject_id=None):
        if subject_id:
            filename = f"{self.model_name}_{subject_id}"
        else:
            filename = f"{self.model_name}_epoch{epoch}"
        
        save_path = os.path.join(self.work_dir, 'models', filename)
        
        self.model.save_weights(f"{save_path}.weights.h5")
        self.logger.info(f"Model weights saved to {save_path}.weights.h5")
        
        try:
            self.model.save(f"{save_path}.keras")
            self.logger.info(f"Full model saved to {save_path}.keras")
        except Exception as e:
            self.logger.warning(f"Failed to save full model: {e}")
        
        if self.save_tflite:
            from utils.tflite_converter import convert_to_tflite
            if convert_to_tflite(self.model, f"{save_path}.tflite"):
                self.logger.info(f"TFLite model saved to {save_path}.tflite")
            else:
                self.logger.warning("TFLite conversion failed, continuing without it")
        
        return save_path
    
    def check_early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.train_loss_history) - 1
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            return False
    
    def train(self, subject_id=None):
        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        
        for epoch in range(self.max_epochs):
            # Train epoch
            _, _ = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(self.val_data, name="val")
            
            # Check early stopping and save best model
            if self.check_early_stopping(val_loss):
                self.logger.info(f"New best model at epoch {epoch+1} with validation loss {val_loss:.4f}")
                self.save_model(epoch, subject_id)
            
            # Stop if early stopping triggered
            if self.early_stop:
                self.logger.info(f"Training stopped at epoch {epoch+1}/{self.max_epochs} due to early stopping")
                break
        
        # Final evaluation on test set if available
        if self.test_data is not None:
            self.logger.info("Evaluating best model on test set")
            test_loss, test_metrics = self.evaluate(self.test_data, name=f"test_{subject_id}" if subject_id else "test")
            
            return {
                'train_history': self.train_loss_history,
                'val_history': self.val_loss_history,
                'best_epoch': self.best_epoch,
                'best_val_loss': self.best_val_loss,
                'test_metrics': test_metrics
            }
        
        return {
            'train_history': self.train_loss_history,
            'val_history': self.val_loss_history,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
