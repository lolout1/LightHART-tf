import os
import time
import datetime
import yaml
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf, UTD_MM_TF
from utils.callbacks import EarlyStoppingTF, CheckpointManagerTF
from utils.metrics import calculate_metrics, BinaryFocalLossTF
from utils.visualization import plot_distribution, plot_loss_curves, plot_confusion_matrix
from utils.common import import_class, save_config

class Trainer:
    """Base trainer class for TensorFlow models."""
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.test_accuracy = 0
        self.test_f1 = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_auc = 0
        self.train_subjects = []
        self.val_subject = None
        self.test_subject = None
        self.optimizer = None
        self.norm_train = None
        self.norm_val = None
        self.norm_test = None
        self.data_loader = {}
        self.early_stop = EarlyStoppingTF(patience=15, min_delta=.001)
        self.inertial_modality = [m for m in arg.dataset_args['modalities'] if m != 'skeleton']
        self.fuse = len(self.inertial_modality) > 1
        
        # Create work directory
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.arg.work_dir)
        
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'
        save_config(arg.config, arg.work_dir)
        
        # Configure GPU
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(self.gpus)} GPU(s), enabled memory growth")
            except: 
                print("Error configuring GPU")
        else:
            print("No GPU found, using CPU")
        
        # Load model
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.model = tf.keras.models.load_model(self.arg.weights)
        
        self.include_val = arg.include_val
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params/(1024 ** 2):.2f} MB')
    
    def add_avg_df(self, results):
        """Add average row to results dataframe."""
        if not results:
            self.print_log("Warning: No results to average")
            return results
        
        avg_dict = {}
        for col in results[0].keys():
            if col != 'test_subject':
                vals = [float(r[col]) for r in results]
                avg_dict[col] = round(sum(vals) / len(vals), 2) if vals else 0
            else:
                avg_dict[col] = 'Average'
        
        results.append(avg_dict)
        return results
    
    def cal_weights(self):
        """Calculate class weights for imbalanced data."""
        labels = self.norm_train['labels']
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        self.pos_weights = num_neg / num_pos if num_pos > 0 else 1.0
    
    def count_parameters(self, model):
        """Count trainable parameters in model."""
        try:
            dummy_input = {'accelerometer': tf.zeros((1, 128, 4))}
            
            try:
                _ = model(dummy_input, training=False)
            except:
                pass
                
            total_params = 0
            for variable in model.trainable_variables:
                total_params += np.prod(variable.shape)
                
            if total_params == 0:
                total_params = model.count_params()
                
            return total_params
        except Exception as e:
            self.print_log(f"Error counting parameters: {e}")
            return 0
    
    def has_empty_value(self, *lists):
        """Check if any lists are empty."""
        return any(len(lst) == 0 for lst in lists)
    
    def load_model(self, model_name, model_args):
        """Load model based on name and arguments."""
        try:
            Model = import_class(model_name)
            model = Model(**model_args)
            dummy_input = {
                'accelerometer': tf.zeros((1, model_args.get('acc_frames', 128), 
                                     model_args.get('acc_coords', 4)))
            }
            
            try:
                _ = model(dummy_input, training=False)
            except:
                pass
                
            param_count = self.count_parameters(model)
            self.print_log(f"Loaded model with {param_count} parameters")
            
            return model
        except Exception as e:
            self.print_log(f"Error loading model {model_name}: {e}")
            import traceback
            self.print_log(traceback.format_exc())
            Model = import_class(model_name)
            return Model(**model_args)
    
    def load_loss(self):
        """Load loss function for training."""
        if self.arg.loss.lower() == 'bce':
            self.criterion = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                label_smoothing=0.0
            )
            if hasattr(self, 'pos_weights') and self.pos_weights > 0:
                self.sample_weight = tf.where(
                    tf.equal(tf.zeros_like(self.norm_train['labels']), 0),
                    tf.ones_like(self.norm_train['labels']) * self.pos_weights,
                    tf.ones_like(self.norm_train['labels'])
                )
        elif self.arg.loss.lower() == 'binary_focal':
            self.criterion = BinaryFocalLossTF(alpha=0.75)
        else:
            self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def load_optimizer(self):
        """Load optimizer for training."""
        if self.arg.optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
        elif self.arg.optimizer.lower() == 'adamw':
            # Try different versions to maintain compatibility
            try:
                self.optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay
                )
            except (ImportError, AttributeError):
                self.print_log("AdamW not available, using Adam with manual weight decay")
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.arg.base_lr
                )
        elif self.arg.optimizer.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.arg.base_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")
    
    def load_data(self):
        """Load and prepare data for training/testing."""
        if self.arg.phase == 'train':
            builder = prepare_smartfallmm_tf(self.arg)
            
            self.norm_train = split_by_subjects_tf(builder, self.train_subjects, self.fuse)
            self.norm_val = split_by_subjects_tf(builder, self.val_subject, self.fuse)
            
            if self.has_empty_value(list(self.norm_val.values())):
                return False
            
            train_dataset = UTD_MM_TF(self.norm_train, self.arg.batch_size)
            val_dataset = UTD_MM_TF(self.norm_val, self.arg.val_batch_size)
            
            self.data_loader['train'] = train_dataset
            self.data_loader['val'] = val_dataset
            
            self.cal_weights()
            
            plot_distribution(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            self.norm_test = split_by_subjects_tf(builder, self.test_subject, self.fuse)
            
            if self.has_empty_value(list(self.norm_test.values())):
                return False
                
            test_dataset = UTD_MM_TF(self.norm_test, self.arg.test_batch_size)
            self.data_loader['test'] = test_dataset
            
            plot_distribution(self.norm_test['labels'], self.arg.work_dir, f'test_{self.test_subject[0]}')
            
            return True
    
    def record_time(self):
        """Record current time."""
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        """Calculate time split."""
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    def print_log(self, string, print_time=True):
        """Print and log to file."""
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)
    
    def create_df(self):
        """Create empty results dataframe."""
        return []
    
    @tf.function
    def train_step(self, data, labels):
        """Single training step with gradient updates."""
        with tf.GradientTape() as tape:
            logits, _ = self.model(data, training=True)
            
            if len(logits.shape) > 1:
                logits = tf.squeeze(logits, axis=-1)
            
            loss = self.criterion(labels, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
        
        return loss, preds
    
    @tf.function
    def test_step(self, data, labels):
        """Single testing/validation step."""
        logits, _ = self.model(data, training=False)
        
        if len(logits.shape) > 1:
            logits = tf.squeeze(logits, axis=-1)
            
        loss = self.criterion(labels, logits)
        preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
        
        return loss, preds
    
    def train(self, epoch):
        """Train model for one epoch."""
        self.model.trainable = True
        self.record_time()
        
        loader = self.data_loader['train']
        timer = {'dataloader': 0.001, 'model': 0.001, 'stats': 0.001}
        
        label_list = []
        pred_list = []
        train_loss = 0
        cnt = 0
        
        for batch_idx, (inputs, targets, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            timer['dataloader'] += self.split_time()
            
            targets = tf.cast(targets, tf.int32)
            
            try:
                loss, preds = self.train_step(inputs, targets)
                
                train_loss += loss
                label_list.extend(targets.numpy())
                pred_list.extend(preds.numpy())
                cnt += 1
            except Exception as e:
                self.print_log(f"Error in training step: {e}")
            
            timer['model'] += self.split_time()
            timer['stats'] += self.split_time()
        
        if cnt > 0:
            train_loss /= cnt
            accuracy, f1, recall, precision, auc_score = calculate_metrics(label_list, pred_list)
            
            self.train_loss_summary.append(train_loss.numpy())
            
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }
            
            self.print_log(
                f'\tTraining Loss: {train_loss.numpy():.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, '
                f'Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%'
            )
            
            self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')
            
            val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
            self.val_loss_summary.append(val_loss)
            
            self.early_stop(val_loss)
    
    def eval(self, epoch, loader_name='val', result_file=None):
        """Evaluate model on validation or test set."""
        self.model.trainable = False
        
        if result_file is not None:
            f_r = open(result_file, 'w')
        
        self.print_log(f'Eval epoch: {epoch+1}')
        
        loss = 0
        cnt = 0
        label_list = []
        pred_list = []
        
        loader = self.data_loader[loader_name]
        
        for batch_idx, (inputs, targets, _) in enumerate(tqdm(loader, desc=f"Evaluating {loader_name}")):
            try:
                targets = tf.cast(targets, tf.int32)
                
                batch_loss, preds = self.test_step(inputs, targets)
                
                loss += batch_loss
                label_list.extend(targets.numpy())
                pred_list.extend(preds.numpy())
                cnt += 1
            except Exception as e:
                self.print_log(f"Error in evaluation step: {e}")
        
        if cnt > 0:
            loss /= cnt
            accuracy, f1, recall, precision, auc_score = calculate_metrics(label_list, pred_list)
            
            if result_file is not None:
                for i, x in enumerate(pred_list):
                    f_r.write(f"{x} ==> {label_list[i]}\n")
                f_r.close()
            
            self.print_log(
                f'{loader_name.capitalize()} Loss: {loss.numpy():.4f}, Acc: {accuracy:.2f}%, '
                f'F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%'
            )
            
            if loader_name == 'val':
                if loss < self.best_loss:
                    self.best_loss = loss
                    ckpt_manager = CheckpointManagerTF(
                        self.model, 
                        self.optimizer, 
                        f'{self.model_path}_{self.test_subject[0]}'
                    )
                    ckpt_manager.save_checkpoint(loss, epoch)
                    self.print_log('Weights Saved')
            else:
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score
            
            return loss.numpy()
        
        return float('inf')
    
    def start(self):
        """Start training/testing process."""
        if self.arg.phase == 'train':
            self.print_log(f'Parameters: \n{yaml.dump(vars(self.arg), default_flow_style=False)}\n')
            
            results = []
            successful_subjects = 0
            
            from utils.common import create_fold_splits
            folds = create_fold_splits(self.arg.subjects, validation_subjects=[38, 46])
            
            for fold in folds:
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                
                self.train_subjects = fold['train']
                self.val_subject = fold['val']
                self.test_subject = fold['test']
                
                self.print_log(f"=== Fold {fold['fold']} ===")
                self.print_log(f"Train subjects: {self.train_subjects}")
                self.print_log(f"Val subjects: {self.val_subject}")
                self.print_log(f"Test subjects: {self.test_subject}")
                
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
                
                if not self.load_data():
                    self.print_log(f"Skipping subject {self.test_subject[0]} due to data loading failure")
                    continue
                
                try:
                    self.load_optimizer()
                    self.load_loss()
                    
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        self.train(epoch)
                        if self.early_stop.early_stop:
                            self.early_stop = EarlyStoppingTF(patience=15, min_delta=.001)
                            break
                    
                    ckpt_manager = CheckpointManagerTF(
                        self.model, 
                        self.optimizer, 
                        f'{self.model_path}_{self.test_subject[0]}'
                    )
                    ckpt_manager.load_best_checkpoint()
                    
                    self.print_log(f' ------------ Test Subject {self.test_subject[0]} -------')
                    self.eval(epoch=0, loader_name='test')
                    self.print_log(f'Test accuracy: {self.test_accuracy}')
                    self.print_log(f'Test F-Score: {self.test_f1}')
                    
                    if self.train_loss_summary and self.val_loss_summary:
                        plot_loss_curves(self.train_loss_summary, self.val_loss_summary, 
                                        self.arg.work_dir, self.test_subject[0])
                    
                    results.append({
                        'test_subject': str(self.test_subject[0]),
                        'accuracy': round(self.test_accuracy, 2),
                        'f1_score': round(self.test_f1, 2),
                        'precision': round(self.test_precision, 2),
                        'recall': round(self.test_recall, 2),
                        'auc': round(self.test_auc, 2)
                    })
                    successful_subjects += 1
                except Exception as e:
                    self.print_log(f"Error during training/testing for subject {self.test_subject[0]}: {e}")
                    import traceback
                    self.print_log(traceback.format_exc())
            
            if successful_subjects > 0:
                results = self.add_avg_df(results)
                
                with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                    json.dump(results, f, indent=4)
            else:
                self.print_log("No subjects completed training successfully. No results to save.")
