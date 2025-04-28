#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Utilities for LightHART-TF

Contains functions for model handling, loading, and saving
"""
import os
import importlib
import traceback
import tensorflow as tf
from utils.tflite_converter import convert_to_tflite

def import_class(import_str):
    """Dynamically import a class"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    
    # Try multiple import paths
    for prefix in ['', 'src.']:
        try:
            module = importlib.import_module(f"{prefix}{mod_str}")
            return getattr(module, class_str)
        except (ImportError, AttributeError):
            continue
            
    raise ImportError(f"Cannot import {class_str} from {mod_str}")

def count_parameters(model):
    """Count trainable parameters in model"""
    total_params = 0
    for var in model.trainable_variables:
        total_params += tf.size(var).numpy()
    return total_params

def load_model(model_name, model_args, dataset_args=None, logger=None):
    """Load and initialize model"""
    try:
        ModelClass = import_class(model_name)
        model = ModelClass(**model_args)
        
        if logger:
            logger(f"Created model: {model_name}")
        
        # Build model with dummy input to initialize weights
        try:
            if dataset_args and 'modalities' in dataset_args and 'accelerometer' in dataset_args['modalities']:
                acc_frames = model_args.get('acc_frames', 128)
                acc_coords = model_args.get('acc_coords', 3)
                
                # Create dummy input with batch size 2
                dummy_input = {
                    'accelerometer': tf.zeros((2, acc_frames, acc_coords), dtype=tf.float32)
                }
                
                # Add skeleton if needed
                if 'skeleton' in dataset_args['modalities']:
                    dummy_input['skeleton'] = tf.zeros((2, acc_frames, 32, 3), dtype=tf.float32)
                
                # Forward pass to build model
                _ = model(dummy_input, training=False)
                
                if logger:
                    logger("Model built successfully")
        except Exception as e:
            if logger:
                logger(f"Warning: Could not pre-build model: {e}")
        
        return model
    except Exception as e:
        if logger:
            logger(f"Error loading model {model_name}: {e}")
        traceback.print_exc()
        raise

def save_model(model, base_filename, logger=None):
    """Save model weights and full model"""
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)
        
        # Save model weights
        weights_path = f"{base_filename}.weights.h5"
        model.save_weights(weights_path)
        if logger:
            logger(f"Saved model weights to {weights_path}")
        
        # Save full model
        try:
            model_path = f"{base_filename}"
            model.save(model_path)
            if logger:
                logger(f"Saved full model to {model_path}")
        except Exception as e:
            if logger:
                logger(f"Warning: Could not save full model: {e}")
        
        # Try to export TFLite model
        try:
            tflite_path = f"{base_filename}.tflite"
            success = convert_to_tflite(
                model=model,
                save_path=tflite_path,
                input_shape=(1, 128, 3)  # Default shape for accelerometer data
            )
            
            if success and logger:
                logger(f"Exported TFLite model to {tflite_path}")
            elif not success and logger:
                logger("Warning: TFLite export failed")
        except Exception as e:
            if logger:
                logger(f"Warning: Could not export TFLite model: {e}")
        
        return True
    except Exception as e:
        if logger:
            logger(f"Error saving model: {e}")
        traceback.print_exc()
        return False

