# src/trainer/cross_validation.py
import os
import logging
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from .train_loop import FallDetectionTrainer
from ..utils.data_processor import prepare_dataset

def create_model_and_optimizer(model_class, model_args, lr=0.001, weight_decay=0.0004):
    """Create model and optimizer"""
    model = model_class(**model_args)
    
    # Create optimizer with weight decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    return model, optimizer

def run_cross_validation(config, all_subjects, model_class, model_args,
                        train_data, val_subjects, work_dir="./experiments"):
    """Run leave-one-subject-out cross-validation"""
    results = []
    
    # Create work directory
    os.makedirs(work_dir, exist_ok=True)
    
    # Loop through test subjects
    for test_subject in all_subjects:
        # Skip validation subjects
        if test_subject in val_subjects:
            continue
        
        logging.info(f"=== Cross-validation fold: Testing on subject {test_subject} ===")
        
        # Create train/test split
        train_subjects = [s for s in all_subjects if s != test_subject and s not in val_subjects]
        
        logging.info(f"Train: {len(train_subjects)} subjects")
        logging.info(f"Val: {len(val_subjects)} subjects")
        logging.info(f"Test: Subject {test_subject}")
        
        # Prepare datasets
        train_ds = prepare_dataset(
            train_data, 
            batch_size=config.batch_size,
            is_training=True
        )
        
        val_ds = prepare_dataset(
            val_data,
            batch_size=config.val_batch_size,
            is_training=False
        )
        
        test_ds = prepare_dataset(
            test_data,
            batch_size=config.test_batch_size,
            is_training=False,
            shuffle=False
        )
        
        # Create model and optimizer
        model, optimizer = create_model_and_optimizer(
            model_class, 
            model_args,
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )
        
        # Create loss function
        if hasattr(config, 'pos_weight') and config.pos_weight is not None:
            loss_fn = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                pos_weight=tf.constant(config.pos_weight)
            )
            class_weights = None
        else:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            class_weights = None
        
        # Create trainer
        trainer = FallDetectionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_ds,
            val_data=val_ds,
            test_data=test_ds,
            patience=15,
            work_dir=work_dir,
            model_name=config.model_saved_name,
            class_weights=class_weights
        )
        
        # Train model
        history = trainer.train(
            epochs=config.num_epoch,
            test_subject=test_subject
        )
        
        # Save results
        result = {
            'test_subject': test_subject,
            'best_epoch': trainer.best_epoch,
            'best_val_loss': trainer.best_val_loss,
            'best_val_f1': trainer.best_val_f1,
            'test_metrics': trainer.evaluate(test_ds, name=f"test_{test_subject}")[1]
        }
        
        results.append(result)
        
        # Save per-subject results
        with open(os.path.join(work_dir, f"results_{test_subject}.json"), 'w') as f:
            json.dump(result, f, indent=2)
    
    # Calculate average results
    avg_metrics = {
        'accuracy': np.mean([r['test_metrics']['accuracy'] for r in results]),
        'f1': np.mean([r['test_metrics']['f1'] for r in results]),
        'precision': np.mean([r['test_metrics']['precision'] for r in results]),
        'recall': np.mean([r['test_metrics']['recall'] for r in results]),
        'auc': np.mean([r['test_metrics']['auc'] for r in results])
    }
    
    results.append({
        'test_subject': 'average',
        'test_metrics': avg_metrics
    })
    
    # Save all results
    with open(os.path.join(work_dir, "all_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create CSV for easy viewing
    df = pd.DataFrame([
        {
            'subject': r['test_subject'],
            'accuracy': r['test_metrics']['accuracy'],
            'f1': r['test_metrics']['f1'],
            'precision': r['test_metrics']['precision'],
            'recall': r['test_metrics']['recall'],
            'auc': r['test_metrics']['auc']
        }
        for r in results
    ])
    
    df.to_csv(os.path.join(work_dir, "results.csv"), index=False)
    
    return results
