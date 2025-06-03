# src/distiller.py
import os
import time
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from train import Trainer, str2bool
from utils.loss import DistillationLoss, BinaryFocalLoss
from models.cross_align_tf import CrossModalAligner

logger = logging.getLogger('distiller')

class Distiller(Trainer):
    def __init__(self, arg):
        super().__init__(arg)
        self.teacher_model = None
        self.cross_aligner = None
        self.distillation_loss = None
        self.criterion = None
        self.print_log("Distiller initialized successfully")
    
    def load_data(self):
        """Override load_data to ensure proper data loading like train.py"""
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
        
        self.print_log("=== Loading Data for Distillation ===")
        self.print_log(f"Train subjects: {self.train_subjects}")
        self.print_log(f"Val subjects: {self.val_subject}")
        self.print_log(f"Test subjects: {self.test_subject}")
        
        try:
            Feeder = self.import_class(self.arg.feeder)
            
            if self.arg.dataset == 'smartfallmm':
                builder = prepare_smartfallmm_tf(self.arg)
            else:
                raise ValueError(f"Unsupported dataset: {self.arg.dataset}")
            
            if self.arg.phase in ['train', 'distill']:
                all_subjects = self.train_subjects + self.val_subject + self.test_subject
                self.print_log(f"Computing global statistics from {len(all_subjects)} subjects")
                
                all_data = split_by_subjects_tf(builder, all_subjects, False, compute_stats_only=True)
                self.acc_mean = all_data.get('acc_mean')
                self.acc_std = all_data.get('acc_std')
                self.skl_mean = all_data.get('skl_mean')
                self.skl_std = all_data.get('skl_std')
                
                self.norm_train = split_by_subjects_tf(builder, self.train_subjects, False, 
                                                      acc_mean=self.acc_mean, acc_std=self.acc_std,
                                                      skl_mean=self.skl_mean, skl_std=self.skl_std)
                
                if not self.norm_train or 'labels' not in self.norm_train or len(self.norm_train['labels']) == 0:
                    self.print_log(f'ERROR: No training data for subjects {self.train_subjects}')
                    return False
                
                self.print_log(f"Training data loaded: {len(self.norm_train['labels'])} samples")
                
                self.norm_val = split_by_subjects_tf(builder, self.val_subject, False,
                                                    acc_mean=self.acc_mean, acc_std=self.acc_std,
                                                    skl_mean=self.skl_mean, skl_std=self.skl_std)
                
                if not self.norm_val or 'labels' not in self.norm_val or len(self.norm_val['labels']) == 0:
                    self.print_log(f'ERROR: No validation data for subjects {self.val_subject}')
                    return False
                
                self.print_log(f"Validation data loaded: {len(self.norm_val['labels'])} samples")
                
                self.norm_test = split_by_subjects_tf(builder, self.test_subject, False,
                                                     acc_mean=self.acc_mean, acc_std=self.acc_std,
                                                     skl_mean=self.skl_mean, skl_std=self.skl_std)
                
                if not self.norm_test or 'labels' not in self.norm_test or len(self.norm_test['labels']) == 0:
                    self.print_log(f'ERROR: No test data for subject {self.test_subject}')
                    return False
                
                self.print_log(f"Test data loaded: {len(self.norm_test['labels'])} samples")
                
                # Analyze age group differences (inherited from parent)
                self.analyze_age_group_differences(self.norm_train, self.norm_val, self.norm_test)
                
                self.pos_weights = self.calculate_class_weights(self.norm_train['labels'])
                
                use_smv = getattr(self.arg, 'use_smv', False)
                window_size = self.arg.dataset_args.get('max_length', 64)
                
                self.print_log(f"Creating data loaders with batch_size={self.arg.batch_size}, use_smv={use_smv}, window_size={window_size}")
                
                self.data_loader['train'] = Feeder(dataset=self.norm_train, batch_size=self.arg.batch_size, 
                                                  use_smv=use_smv, window_size=window_size)
                self.data_loader['val'] = Feeder(dataset=self.norm_val, batch_size=self.arg.val_batch_size,
                                                use_smv=use_smv, window_size=window_size)
                self.data_loader['test'] = Feeder(dataset=self.norm_test, batch_size=self.arg.test_batch_size,
                                                 use_smv=use_smv, window_size=window_size)
                
                self.print_log(f"Train batches: {len(self.data_loader['train'])}")
                self.print_log(f"Val batches: {len(self.data_loader['val'])}")
                self.print_log(f"Test batches: {len(self.data_loader['test'])}")
                
                self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, f'train_s{self.test_subject[0]}')
                self.distribution_viz(self.norm_val['labels'], self.arg.work_dir, f'val_s{self.test_subject[0]}')
                self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_s{self.test_subject[0]}')
                
                self.print_log("=== Data Loading Complete ===")
                return True
                
        except Exception as e:
            self.print_log(f"ERROR in load_data: {e}")
            import traceback
            self.print_log(traceback.format_exc())
            return False
    
    def load_loss(self):
        """Override to load both standard loss and distillation loss"""
        from utils.loss import BinaryFocalLoss
        self.pos_weights = getattr(self, 'pos_weights', tf.constant(1.0))
        self.criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.print_log(f"Loaded standard loss with pos_weight: {self.pos_weights.numpy()}")
        self.load_distillation_loss()
    
    def load_distillation_loss(self):
        """Load distillation loss with proper configuration"""
        temperature = getattr(self.arg, 'temperature', 4.5)
        alpha = getattr(self.arg, 'alpha', 0.6)
        pos_weight = self.pos_weights if hasattr(self, 'pos_weights') else None
        
        self.distillation_loss = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            pos_weight=pos_weight
        )
        
        self.print_log(f"Initialized distillation loss with temp={temperature}, alpha={alpha}")
    
    def load_teacher_model(self):
        """Load teacher model - match PyTorch implementation"""
        logger.info(f"Loading teacher model: {self.arg.teacher_model}")
        
        teacher_class = self.import_class(self.arg.teacher_model)
        teacher_model = teacher_class(**self.arg.teacher_args)
        
        acc_frames = self.arg.teacher_args.get('acc_frames', 64)
        acc_coords = self.arg.teacher_args.get('acc_coords', 3)
        mocap_frames = self.arg.teacher_args.get('mocap_frames', 64)
        num_joints = self.arg.teacher_args.get('num_joints', 32)
        
        dummy_input = {
            'accelerometer': tf.zeros((1, acc_frames, acc_coords)),
            'skeleton': tf.zeros((1, mocap_frames, num_joints, 3))
        }
        _ = teacher_model(dummy_input, training=False)
        
        subject_id = self.test_subject[0] if hasattr(self, 'test_subject') and self.test_subject else None
        
        if subject_id:
            weight_paths = [
                f"{self.arg.teacher_weight}_{subject_id}.weights.h5",
                f"{self.arg.teacher_weight}_{subject_id}.keras",
                f"{os.path.dirname(self.arg.teacher_weight)}/teacher_model_{subject_id}.weights.h5",
                f"{os.path.dirname(self.arg.teacher_weight)}/teacher_model_{subject_id}.keras"
            ]
            
            loaded = False
            for weight_path in weight_paths:
                if os.path.exists(weight_path):
                    try:
                        if weight_path.endswith('.weights.h5'):
                            teacher_model.load_weights(weight_path)
                        else:
                            teacher_model = tf.keras.models.load_model(weight_path, compile=False)
                        logger.info(f"Loaded teacher weights from {weight_path}")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load weights from {weight_path}: {e}")
            
            if not loaded:
                logger.error(f"No teacher weights found for subject {subject_id}")
        
        teacher_model.trainable = False
        return teacher_model
    
    def load_cross_aligner(self):
        """Load cross-modal aligner"""
        feature_dim = self.arg.model_args.get('embed_dim', 32)
        num_heads = self.arg.model_args.get('num_heads', 4)
        aligner = CrossModalAligner(feature_dim=feature_dim, num_heads=num_heads)
        logger.info(f"Initialized cross-modal aligner with dim={feature_dim}, heads={num_heads}")
        return aligner
    
    def viz_feature(self, teacher_features, student_features, epoch):
        """Visualize feature distributions - match PyTorch implementation"""
        try:
            teacher_features = tf.reshape(teacher_features, (teacher_features.shape[0], -1))
            student_features = tf.reshape(student_features, (student_features.shape[0], -1))
            
            plt.figure(figsize=(12, 6))
            
            num_samples = min(8, teacher_features.shape[0])
            for i in range(num_samples):
                plt.subplot(2, 4, i+1)
                
                sns.kdeplot(teacher_features[i, :].numpy(), bw_adjust=0.5, 
                           color='blue', label='Teacher')
                sns.kdeplot(student_features[i, :].numpy(), bw_adjust=0.5,
                           color='red', label='Student')
                
                plt.legend()
                plt.title(f'Sample {i+1}')
                
            plt.tight_layout()
            viz_path = os.path.join(self.arg.work_dir, 'visualizations', 
                                   f'feature_KDE_{self.test_subject[0]}_epoch{epoch}.png')
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            plt.savefig(viz_path)
            plt.close()
        except Exception as e:
            self.print_log(f"Error in feature visualization: {e}")
    
    def distill_step(self, inputs, targets):
        """Single distillation training step with gradient tape"""
        with tf.GradientTape() as tape:
            teacher_outputs = self.teacher_model(inputs, training=False)
            student_outputs = self.model(inputs, training=True)
            
            if isinstance(teacher_outputs, tuple) and len(teacher_outputs) > 1:
                teacher_logits, teacher_features = teacher_outputs
            else:
                teacher_logits = teacher_outputs
                teacher_features = None
            
            if isinstance(student_outputs, tuple) and len(student_outputs) > 1:
                student_logits, student_features = student_outputs
            else:
                student_logits = student_outputs
                student_features = None
            
            if self.cross_aligner and teacher_features is not None and student_features is not None:
                aligned_features = self.cross_aligner(student_features, teacher_features)
            else:
                aligned_features = student_features
            
            loss = self.distillation_loss(
                student_logits, teacher_logits, targets,
                teacher_features, aligned_features
            )
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, student_logits
    
    def train(self, epoch):
        """Training loop for distillation - match train.py structure"""
        self.print_log(f'Starting Distillation Epoch: {epoch+1}/{self.arg.num_epoch}')
        
        loader = self.data_loader['train']
        train_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        
        start_time = time.time()
        
        from tqdm import tqdm
        progress = tqdm(range(len(loader)), ncols=80, desc=f'Distill epoch {epoch+1}')
        
        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                
                loss, student_logits = self.distill_step(inputs, targets)
                
                probabilities = tf.sigmoid(student_logits)
                predictions = tf.cast(probabilities > 0.5, tf.int32)
                
                train_loss += loss.numpy()
                all_labels.extend(targets.numpy().flatten())
                all_preds.extend(predictions.numpy().flatten())
                all_probs.extend(probabilities.numpy().flatten())
                steps += 1
                
                progress.set_postfix({'loss': f'{loss.numpy():.4f}'})
                
                if epoch % 10 == 0 and batch_idx == 0:
                    teacher_outputs = self.teacher_model(inputs, training=False)
                    student_outputs = self.model(inputs, training=False)
                    
                    if isinstance(teacher_outputs, tuple) and isinstance(student_outputs, tuple):
                        if len(teacher_outputs) > 1 and len(student_outputs) > 1:
                            _, teacher_features = teacher_outputs
                            _, student_features = student_outputs
                            self.viz_feature(teacher_features, student_features, epoch)
                
            except Exception as e:
                self.print_log(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if steps > 0:
            train_loss /= steps
            train_time = time.time() - start_time
            
            accuracy, f1, recall, precision, auc_score = self.calculate_metrics(
                all_labels, all_preds, all_probs
            )
            
            self.train_loss_summary.append(float(train_loss))
            
            self.print_log(f'Epoch {epoch+1} Distillation Results:')
            self.print_log(f'  Loss: {train_loss:.4f}')
            self.print_log(f'  Accuracy: {accuracy:.2f}%')
            self.print_log(f'  F1 Score: {f1:.2f}%')  
            self.print_log(f'  Precision: {precision:.2f}%')
            self.print_log(f'  Recall: {recall:.2f}%')
            self.print_log(f'  AUC: {auc_score:.2f}%')
            self.print_log(f'  Time: {train_time:.2f}s')
            self.print_log(f'  Batches: {steps}/{len(loader)}')
        else:
            self.print_log("Warning: No valid training batches!")
            return True
        
        val_loss = self.eval(epoch, loader_name='val')
        self.val_loss_summary.append(float(val_loss))
        
        self.early_stop(val_loss)
        if self.early_stop.early_stop:
            self.print_log(f"Early stopping triggered at epoch {epoch+1}")
            return True
        
        return False
    
    def eval(self, epoch, loader_name='val', result_file=None):
        """Override eval to ensure correct model saving based on lowest val loss"""
        self.print_log(f'Evaluating {loader_name} at epoch {epoch+1}')
        loader = self.data_loader[loader_name]
        eval_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        steps = 0
        
        from tqdm import tqdm
        progress = tqdm(range(len(loader)), ncols=80, desc=f'{loader_name.capitalize()}')
        
        for batch_idx in progress:
            try:
                inputs, targets, _ = loader[batch_idx]
                targets = tf.cast(targets, tf.float32)
                
                outputs = self.model(inputs, training=False)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(targets, logits)
                probabilities = tf.sigmoid(logits)
                predictions = tf.cast(probabilities > 0.5, tf.int32)
                
                eval_loss += loss.numpy()
                all_labels.extend(targets.numpy().flatten())
                all_preds.extend(predictions.numpy().flatten())
                all_probs.extend(probabilities.numpy().flatten())
                steps += 1
                
                progress.set_postfix({'loss': f'{loss.numpy():.4f}'})
            except Exception as e:
                self.print_log(f"Error in evaluation batch {batch_idx}: {e}")
                continue
        
        if steps > 0:
            eval_loss /= steps
            accuracy, f1, recall, precision, auc_score = self.calculate_metrics(all_labels, all_preds, all_probs)
            
            self.print_log(f'{loader_name.capitalize()} Results:')
            self.print_log(f'  Loss: {eval_loss:.4f}')
            self.print_log(f'  Accuracy: {accuracy:.2f}%')
            self.print_log(f'  F1 Score: {f1:.2f}%')
            self.print_log(f'  Precision: {precision:.2f}%')
            self.print_log(f'  Recall: {recall:.2f}%')
            self.print_log(f'  AUC: {auc_score:.2f}%')
            self.print_log(f'  Batches: {steps}/{len(loader)}')
            
            # Save model if validation loss improved
            if loader_name == 'val':
                if eval_loss < self.best_loss:
                    self.print_log(f'New best validation loss: {eval_loss:.4f} (previous: {self.best_loss:.4f})')
                    self.best_loss = eval_loss
                    self.save_model()
                    self.print_log(f'✓ Model saved with best validation loss: {eval_loss:.4f}')
                else:
                    self.print_log(f'Validation loss: {eval_loss:.4f} (best: {self.best_loss:.4f}) - not saving')
        else:
            self.print_log(f"Warning: No valid {loader_name} batches!")
            return float('inf')
        
        if result_file is not None:
            with open(result_file, 'w') as f:
                f.write(f"Predictions for {loader_name} epoch {epoch+1}\n")
                f.write("true,predicted,probability\n")
                for true, pred, prob in zip(all_labels, all_preds, all_probs):
                    f.write(f'{true},{pred},{prob:.4f}\n')
        
        if loader_name == 'test':
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
            self.cm_viz(all_preds, all_labels)
        
        return eval_loss
    
    def start(self):
        """Main distillation training loop with older subjects integration"""
        if self.arg.phase in ['distill', 'train']:
            self.print_log('=== Starting Knowledge Distillation ===')
            self.print_log('Configuration:')
            self.print_log(yaml.dump(vars(self.arg), default_flow_style=False))
            
            # Safety checks for older subjects
            if hasattr(self, 'test_eligible_subjects'):
                older_in_test = [s for s in self.test_eligible_subjects if s < 29]
                if older_in_test:
                    raise ValueError(f"ERROR: Older subjects {older_in_test} found in test eligible set!")
            
            if hasattr(self, 'fixed_val_subjects'):
                older_in_val = [s for s in self.fixed_val_subjects if s < 29]
                if older_in_val:
                    self.print_log(f"WARNING: Removing older subjects {older_in_val} from validation set")
                    self.fixed_val_subjects = [s for s in self.fixed_val_subjects if s >= 29]
            
            results = []
            
            for fold_idx, test_subject in enumerate(self.test_eligible_subjects):
                if test_subject < 29:
                    self.print_log(f"ERROR: Attempting to test on older subject {test_subject}")
                    continue
                
                fold_start_time = time.time()
                
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')  # Reset for each fold
                
                self.test_subject = [test_subject]
                self.val_subject = self.fixed_val_subjects
                remaining_eligible = [s for s in self.test_eligible_subjects if s != test_subject]
                self.train_subjects = self.fixed_train_subjects + remaining_eligible
                
                self.print_log(f'\n{"="*60}')
                self.print_log(f'DISTILLATION FOLD {fold_idx+1}/{self.total_folds}: Test Subject {test_subject}')
                self.print_log(f'{"="*60}')
                
                # Display subject allocation
                self.print_log(f'\n📊 SUBJECT ALLOCATION FOR FOLD {fold_idx+1}:')
                self.print_log(f'───────────────────────────────────────')
                self.print_log(f'TRAIN SUBJECTS ({len(self.train_subjects)} total):')
                older_train = sorted([s for s in self.train_subjects if s < 29])
                younger_train = sorted([s for s in self.train_subjects if s >= 29])
                if older_train:
                    self.print_log(f'  Older (1-28): {older_train} ({len(older_train)} subjects)')
                self.print_log(f'  Younger (29+): {younger_train} ({len(younger_train)} subjects)')
                self.print_log(f'\nVAL SUBJECTS ({len(self.val_subject)} total):')
                self.print_log(f'  {sorted(self.val_subject)}')
                self.print_log(f'\nTEST SUBJECTS ({len(self.test_subject)} total):')
                self.print_log(f'  {self.test_subject}')
                self.print_log(f'───────────────────────────────────────\n')
                
                try:
                    tf.keras.backend.clear_session()
                    
                    self.teacher_model = self.load_teacher_model()
                    self.model = self.load_model()
                    self.print_log(f'Student Model Parameters: {self.count_parameters()}')
                    
                    if hasattr(self.arg, 'use_cross_aligner') and self.arg.use_cross_aligner:
                        self.cross_aligner = self.load_cross_aligner()
                    
                    if not self.load_data():
                        self.print_log(f"Failed to load data for subject {test_subject}")
                        continue
                    
                    self.load_optimizer()
                    self.load_loss()
                    
                    self.early_stop.reset()
                    
                    # Training loop
                    best_epoch = 0
                    for epoch in range(self.arg.num_epoch):
                        if self.train(epoch):
                            break
                        if self.val_loss_summary[-1] == self.best_loss:
                            best_epoch = epoch + 1
                    
                    self.print_log(f'\nTraining completed. Best model saved at epoch {best_epoch} with val_loss={self.best_loss:.4f}')
                    
                    # Reload best model and evaluate
                    self.model = self.load_model()
                    self.load_weights()
                    
                    self.print_log(f'\n=== Testing Subject {self.test_subject[0]} with Best Model ===')
                    self.eval(epoch=0, loader_name='test')
                    
                    self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    
                    subject_result = {
                        'test_subject': str(self.test_subject[0]),
                        'accuracy': round(self.test_accuracy, 2),
                        'f1_score': round(self.test_f1, 2),
                        'precision': round(self.test_precision, 2),
                        'recall': round(self.test_recall, 2),
                        'auc': round(self.test_auc, 2),
                        'best_val_loss': round(self.best_loss, 4),
                        'best_epoch': best_epoch,
                        'fold_time': round(time.time() - fold_start_time, 2)
                    }
                    results.append(subject_result)
                    
                    pd.DataFrame(results).to_csv(
                        os.path.join(self.arg.work_dir, 'interim_distillation_results.csv'), 
                        index=False
                    )
                    
                    self.print_log(f'\nFold {fold_idx+1} completed in {subject_result["fold_time"]:.2f}s')
                    self.print_log(f'Results: Acc={self.test_accuracy:.2f}%, F1={self.test_f1:.2f}%')
                    
                except Exception as e:
                    self.print_log(f"Error in fold {fold_idx+1}: {e}")
                    import traceback
                    self.print_log(traceback.format_exc())
                    continue
            
            # Save final results
            if results:
                df_results = pd.DataFrame(results)
                
                stats = df_results.describe().round(2)
                
                avg_row = df_results.mean(numeric_only=True).round(2)
                avg_row['test_subject'] = 'Average'
                df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
                
                df_results.to_csv(os.path.join(self.arg.work_dir, 'distillation_results.csv'), index=False)
                stats.to_csv(os.path.join(self.arg.work_dir, 'distillation_statistics.csv'))
                
                self.print_log("\n" + "="*60)
                self.print_log("DISTILLATION RESULTS SUMMARY")
                self.print_log("="*60)
                self.print_log(df_results.to_string(index=False))
                self.print_log("\nStatistics:")
                self.print_log(stats.to_string())
                self.print_log("="*60)
                
                self.create_overall_visualization(df_results)
                self.create_summary_report(df_results, stats)
            else:
                self.print_log("Warning: No results collected!")
        else:
            self.print_log(f"Phase {self.arg.phase} not implemented")

def get_distill_args():
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    
    parser.add_argument('--config', default='./config/smartfallmm/distill.yaml')
    parser.add_argument('--dataset', type=str, default='smartfallmm')
    
    # Training args
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=16)
    parser.add_argument('--num-epoch', type=int, default=80)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0004)
    
    # Model args
    parser.add_argument('--model', default=None)
    parser.add_argument('--device', nargs='+', default=[0], type=int)
    parser.add_argument('--model-args', default=None, type=str)
    parser.add_argument('--model-saved-name', type=str, default='distilled_model')
    
    # Teacher model args
    parser.add_argument('--teacher-model', type=str, default=None)
    parser.add_argument('--teacher-args', type=str, default=None)
    parser.add_argument('--teacher-weight', type=str, default=None)
    
    # Distillation args
    parser.add_argument('--temperature', type=float, default=4.5)
    parser.add_argument('--alpha', type=float, default=0.6) 
    parser.add_argument('--use-cross-aligner', type=str2bool, default=False)
    
    # Dataset args
    parser.add_argument('--dataset-args', default=None, type=str)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None)
    
    # Other args
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--work-dir', type=str, default='distilled')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='distill')
    parser.add_argument('--num-worker', type=int, default=22)
    parser.add_argument('--use-smv', type=str2bool, default=False)
    
    # Cross validation args
    parser.add_argument('--train-subjects-fixed', nargs='+', type=int)
    parser.add_argument('--val-subjects-fixed', nargs='+', type=int)
    parser.add_argument('--test-eligible-subjects', nargs='+', type=int)
    parser.add_argument('--include-older-subjects', type=str2bool, default=True)
    
    return parser

def main():
    parser = get_distill_args()
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                if not hasattr(args, k) or getattr(args, k) is None:
                    setattr(args, k, v)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device[0])
    
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    distiller = Distiller(args)
    distiller.start()

if __name__ == "__main__":
    main()
