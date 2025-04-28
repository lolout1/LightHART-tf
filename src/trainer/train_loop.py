#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import numpy as np
from utils.metrics import calculate_metrics

def train_step(model, inputs, targets, optimizer, criterion):
    """Execute a single training step and return loss and predictions"""
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(inputs, training=True)
        
        # Extract logits (model may return (logits, features))
        if isinstance(outputs, tuple) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = outputs
        
        # Compute loss based on output shape
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            # Multi-class case
            loss = criterion(targets, logits)
        else:
            # Binary case - ensure proper shape
            loss = criterion(targets, tf.squeeze(logits))
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply gradients with clipping to prevent explosion
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Calculate predictions
    if len(logits.shape) > 1 and logits.shape[-1] > 1:
        # Multi-class case
        predictions = tf.argmax(logits, axis=-1)
    else:
        # Binary case
        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    
    return loss, predictions

def eval_step(model, inputs, targets, criterion):
    """Execute a single evaluation step and return loss and predictions"""
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Extract logits (model may return (logits, features))
    if isinstance(outputs, tuple) and len(outputs) > 0:
        logits = outputs[0]
    else:
        logits = outputs
    
    # Compute loss based on output shape
    if len(logits.shape) > 1 and logits.shape[-1] > 1:
        # Multi-class case
        loss = criterion(targets, logits)
    else:
        # Binary case - ensure proper shape
        loss = criterion(targets, tf.squeeze(logits))
    
    # Calculate predictions
    if len(logits.shape) > 1 and logits.shape[-1] > 1:
        # Multi-class case
        predictions = tf.argmax(logits, axis=-1)
    else:
        # Binary case
        predictions = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
    
    return loss, predictions, logits

def train_epoch(model, optimizer, criterion, data_loader, epoch, num_epochs, logger=None):
    """Train model for one epoch and return loss and metrics"""
    model.trainable = True
    
    # Record start time
    start_time = time.time()
    
    # Initialize metrics
    train_loss = 0.0
    all_labels = []
    all_preds = []
    batch_count = 0
    
    # Get total number of batches
    total_batches = len(data_loader)
    
    # Skip tqdm and implement custom progress reporting
    for batch_idx, (inputs, targets, _) in enumerate(data_loader):
        # Log progress every 10 batches
        if batch_idx % 10 == 0 and logger:
            logger(f"Training epoch {epoch+1}/{num_epochs}: batch {batch_idx+1}/{total_batches}")
            
        # Convert targets to float32
        targets = tf.cast(targets, tf.float32)
        
        # Train step
        loss, predictions = train_step(model, inputs, targets, optimizer, criterion)
        
        # Update metrics
        train_loss += loss.numpy()
        all_labels.extend(targets.numpy())
        all_preds.extend(predictions.numpy())
        batch_count += 1
    
    # Calculate average loss and metrics
    train_loss /= batch_count
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Log completion
    epoch_time = time.time() - start_time
    if logger:
        logger(
            f"Epoch {epoch+1}/{num_epochs} complete: "
            f"Loss={train_loss:.4f}, "
            f"Acc={metrics['accuracy']:.2f}%, "
            f"F1={metrics['f1']:.2f}%, "
            f"Prec={metrics['precision']:.2f}%, "
            f"Rec={metrics['recall']:.2f}%, "
            f"AUC={metrics['auc']:.2f}% "
            f"({epoch_time:.2f}s)"
        )
    
    return train_loss, metrics, all_labels, all_preds

def evaluate_model(model, criterion, data_loader, epoch=0, name='val', logger=None):
    """Evaluate model on a dataset and return loss and metrics"""
    model.trainable = False
    
    # Record start time
    start_time = time.time()
    
    # Initialize metrics
    eval_loss = 0.0
    all_labels = []
    all_preds = []
    batch_count = 0
    
    # Get total number of batches
    total_batches = len(data_loader)
    
    # Skip tqdm and implement custom progress reporting
    for batch_idx, (inputs, targets, _) in enumerate(data_loader):
        # Log progress every 5 batches
        if batch_idx % 5 == 0 and logger:
            logger(f"Evaluating {name} (epoch {epoch+1}): batch {batch_idx+1}/{total_batches}")
            
        # Convert targets to float32
        targets = tf.cast(targets, tf.float32)
        
        # Evaluation step
        loss, predictions, _ = eval_step(model, inputs, targets, criterion)
        
        # Update metrics
        eval_loss += loss.numpy()
        all_labels.extend(targets.numpy())
        all_preds.extend(predictions.numpy())
        batch_count += 1
    
    # Calculate average loss and metrics
    eval_loss /= batch_count
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Log completion
    eval_time = time.time() - start_time
    if logger:
        logger(
            f"{name.capitalize()} evaluation complete: "
            f"Loss={eval_loss:.4f}, "
            f"Acc={metrics['accuracy']:.2f}%, "
            f"F1={metrics['f1']:.2f}%, "
            f"Prec={metrics['precision']:.2f}%, "
            f"Rec={metrics['recall']:.2f}%, "
            f"AUC={metrics['auc']:.2f}% "
            f"({eval_time:.2f}s)"
        )
    
    return eval_loss, metrics
