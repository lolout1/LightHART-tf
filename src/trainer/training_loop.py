#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Loop Module for LightHART-TF

Contains training and evaluation functions for fall detection models.
"""
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.metrics import calculate_metrics

@tf.function
def train_step(model, inputs, targets, optimizer, criterion):
    """Training step with gradient tape"""
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(inputs, training=True)
        
        # Extract logits (model may return (logits, features))
        if isinstance(outputs, tuple) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = outputs
        
        # Reshape logits if needed
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            # Multi-class case
            pass
        else:
            # Binary case
            logits = tf.squeeze(logits)
        
        # Compute loss
        loss = criterion(targets, logits)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Clip gradients to prevent explosion
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Calculate predictions
    predictions = cal_prediction(logits)
    
    return loss, predictions

@tf.function
def eval_step(model, inputs, targets, criterion):
    """Evaluation step"""
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Extract logits (model may return (logits, features))
    if isinstance(outputs, tuple) and len(outputs) > 0:
        logits = outputs[0]
    else:
        logits = outputs
    
    # Reshape logits if needed
    if len(logits.shape) > 1 and logits.shape[-1] > 1:
        # Multi-class case
        pass
    else:
        # Binary case
        logits = tf.squeeze(logits)
    
    # Compute loss
    loss = criterion(targets, logits)
    
    # Calculate predictions
    predictions = cal_prediction(logits)
    
    return loss, predictions, logits

def cal_prediction(logits):
    """Calculate binary predictions from logits"""
    # Handle different output shapes
    if isinstance(logits, tuple) and len(logits) > 0:
        logits = logits[0]  # Sometimes model returns (logits, features)
        
    if len(logits.shape) > 1 and logits.shape[-1] > 1:
        # Multi-class case
        return tf.argmax(logits, axis=-1)
    else:
        # Binary case
        return tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
# In trainer/training_loop.py - train_epoch function
def train_epoch(model, optimizer, criterion, data_loader, epoch, num_epochs, logger):
    """Train model for one epoch"""
    model.trainable = True

    # Start timer
    start_time = time.time()

    # Initialize metrics
    train_loss = 0.0
    all_labels = []
    all_preds = []
    steps = 0

    # Get total number of batches
    total_steps = len(data_loader)

    # Always print beginning of epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Training started ({total_steps} batches)")
    logger(f"Epoch {epoch+1}/{num_epochs} - Training started ({total_steps} batches)")

    # Create progress bar with correct total
    desc = f"Epoch {epoch+1}/{num_epochs}"
    progress_bar = tqdm(data_loader, desc=desc, total=total_steps)

    # Iterate through batches
    for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
        try:
            # Convert targets to float32
            targets = tf.cast(targets, tf.float32)

            # Train step
            loss, predictions = train_step(model, inputs, targets, optimizer, criterion)

            # Update metrics
            loss_val = loss.numpy() if isinstance(loss, tf.Tensor) else float(loss)
            train_loss += loss_val
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            steps += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{train_loss/steps:.4f}"
            })

            # Log progress periodically
            if batch_idx % 10 == 0:
                batch_msg = f"Batch {batch_idx+1}/{total_steps}, Loss: {loss_val:.4f}"
                print(batch_msg)
                logger(batch_msg)

            # Break if we've gone through all batches
            if batch_idx >= total_steps - 1:
                break

        except tf.errors.InvalidArgumentError as e:
            error_msg = f"Error in batch {batch_idx}: {e}"
            print(error_msg)
            logger(error_msg)
            # Continue with next batch
            continue

    # Calculate average loss and metrics
    train_loss /= steps if steps > 0 else 1.0
    metrics = calculate_metrics(all_labels, all_preds)

    # Calculate epoch time
    epoch_time = time.time() - start_time

    # Log results
    results_msg = (
        f"Epoch {epoch+1}: "
        f"Train Loss={train_loss:.4f}, "
        f"Acc={metrics['accuracy']:.2f}%, "
        f"F1={metrics['f1']:.2f}%, "
        f"Prec={metrics['precision']:.2f}%, "
        f"Rec={metrics['recall']:.2f}%, "
        f"AUC={metrics['auc']:.2f}% "
        f"({epoch_time:.2f}s)"
    )
    print(results_msg)
    logger(results_msg)

    return train_loss, metrics, all_labels, all_preds

def evaluate_model(model, criterion, data_loader, epoch, name='val', logger=None):
    """Evaluate model on a dataset"""
    model.trainable = False

    # Start timer
    start_time = time.time()

    # Initialize metrics
    eval_loss = 0.0
    all_labels = []
    all_preds = []
    all_logits = []
    steps = 0

    # Get total number of batches
    total_steps = len(data_loader)

    # Always print beginning of evaluation
    eval_start_msg = f"Evaluating {name} (epoch {epoch+1}) - {total_steps} batches"
    print(eval_start_msg)
    if logger:
        logger(eval_start_msg)

    # Create progress bar with correct total
    desc = f"Eval {name} ({epoch+1})"
    progress_bar = tqdm(data_loader, desc=desc, total=total_steps)

    # Iterate through batches
    for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
        try:
            # Convert targets to float32
            targets = tf.cast(targets, tf.float32)

            # Evaluation step
            loss, predictions, logits = eval_step(model, inputs, targets, criterion)

            # Update metrics
            loss_val = loss.numpy() if isinstance(loss, tf.Tensor) else float(loss)
            eval_loss += loss_val
            all_labels.extend(targets.numpy())
            all_preds.extend(predictions.numpy())
            all_logits.extend(logits.numpy())
            steps += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{eval_loss/steps:.4f}"
            })

            # Log progress periodically
            if batch_idx % 5 == 0:
                batch_msg = f"Evaluating {name} - Batch {batch_idx+1}/{total_steps}, Loss: {loss_val:.4f}"
                print(batch_msg)
                if logger:
                    logger(batch_msg)

            # Break if we've gone through all batches
            if batch_idx >= total_steps - 1:
                break

        except tf.errors.InvalidArgumentError as e:
            error_msg = f"Error in batch {batch_idx}: {e}"
            print(error_msg)
            if logger:
                logger(error_msg)
            # Continue with next batch
            continue

    # Calculate average loss and metrics
    eval_loss /= steps if steps > 0 else 1.0
    metrics = calculate_metrics(all_labels, all_preds)

    # Calculate epoch time
    epoch_time = time.time() - start_time

    # Log results - always print even if logger is None
    results_msg = (
        f"{name.capitalize()} evaluation complete: "
        f"Loss={eval_loss:.4f}, "
        f"Acc={metrics['accuracy']:.2f}%, "
        f"F1={metrics['f1']:.2f}%, "
        f"Prec={metrics['precision']:.2f}%, "
        f"Rec={metrics['recall']:.2f}%, "
        f"AUC={metrics['auc']:.2f}% "
        f"({epoch_time:.2f}s)"
    )
    print(results_msg)
    if logger:
        logger(results_msg)

    return eval_loss, metrics
