import os
import time
import datetime
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import traceback
import sys
import logging

class Trainer:
    def __init__(self, arg):
        self.arg = arg
        self.train_loss_summary = []
        self.val_loss_summary = []
        self.best_loss = float('inf')
        self.best_f1 = 0.0
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
        
        if not hasattr(self.arg, 'batch_size') or not self.arg.batch_size:
            self.arg.batch_size = 32
        if not hasattr(self.arg, 'test_batch_size') or not self.arg.test_batch_size:
            self.arg.test_batch_size = 32
        if not hasattr(self.arg, 'val_batch_size') or not self.arg.val_batch_size:
            self.arg.val_batch_size = 32
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        from utils.common import import_class
        from utils.callbacks import EarlyStoppingTF
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf, UTD_MM_TF
        from utils.metrics import calculate_metrics
        
        self.import_class = import_class
        self.EarlyStoppingTF = EarlyStoppingTF
        self.prepare_smartfallmm_tf = prepare_smartfallmm_tf
        self.split_by_subjects_tf = split_by_subjects_tf
        self.UTD_MM_TF = UTD_MM_TF
        self.calculate_metrics = calculate_metrics
        
        self.early_stop = self.EarlyStoppingTF(patience=15, min_delta=.001)
        self.inertial_modality = [m for m in arg.dataset_args['modalities'] if m != 'skeleton']
        self.fuse = len(self.inertial_modality) > 1
        
        self.project_root = self.get_project_root()
        
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.arg.work_dir, exist_ok=True)
        
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'
        os.makedirs(os.path.join(self.arg.work_dir, 'models'), exist_ok=True)
        
        self.save_config(arg.config, arg.work_dir)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Found {len(gpus)} GPU(s), enabled memory growth")
            except: 
                logging.warning("Error configuring GPU")
        
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            try:
                self.model = self.load_model(arg.model, arg.model_args)
                if self.arg.weights.endswith('.h5'):
                    self.model.load_weights(self.arg.weights)
                else:
                    self.model = tf.keras.models.load_model(self.arg.weights)
            except Exception as e:
                self.print_log(f"Error loading model: {e}")
                self.print_log(traceback.format_exc())
                raise
        
        self.include_val = arg.include_val
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params/(1024 ** 2):.2f} MB')
    
    def get_project_root(self):
        cwd = os.getcwd()
        if os.path.basename(cwd) == 'scripts':
            return os.path.dirname(cwd)
        return cwd
    
    def save_config(self, src_path, desc_path):
        config_filename = os.path.basename(src_path)
        with open(src_path, 'r') as f_src:
            with open(f'{desc_path}/{config_filename}', 'w') as f_dst:
                f_dst.write(f_src.read())
    
    def cal_weights(self):
        labels = self.norm_train['labels']
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        self.pos_weights = num_neg / num_pos if num_pos > 0 else 1.0
    
    def count_parameters(self, model):
        total_params = 0
        try:
            for variable in model.trainable_variables:
                total_params += np.prod(variable.shape)
            if total_params == 0:
                for variable in model.trainable_weights:
                    total_params += np.prod(variable.shape)
        except:
            try:
                total_params = model.count_params()
            except:
                pass
        return total_params
    
    def has_empty_value(self, *lists):
        return any(not lst or len(lst) == 0 for lst in lists)
    
    def load_model(self, model_name, model_args):
        try:
            ModelClass = self.import_class(model_name)
            model = ModelClass(**model_args)
            sample_input = tf.zeros((1, model_args.get('acc_frames', 128), model_args.get('acc_coords', 4)))
            model(sample_input)
            return model
        except Exception as e:
            self.print_log(f"Error loading model: {e}")
            self.print_log(traceback.format_exc())
            raise
    
    def load_loss(self):
        if self.arg.loss.lower() == 'bce':
            self.criterion = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )
        elif self.arg.loss.lower() == 'binary_focal':
            class BinaryFocalLoss(tf.keras.losses.Loss):
                def __init__(self, alpha=0.75, gamma=2.0):
                    super(BinaryFocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                
                def call(self, y_true, y_pred):
                    y_true = tf.cast(y_true, tf.float32)
                    prob = tf.sigmoid(y_pred)
                    pt = tf.where(tf.equal(y_true, 1.0), prob, 1 - prob)
                    alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
                    focal_loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
                    return focal_loss
            self.criterion = BinaryFocalLoss(alpha=0.75)
        else:
            self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def load_optimizer(self):
        if self.arg.optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
        elif self.arg.optimizer.lower() == 'adamw':
            try:
                self.optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=self.arg.base_lr,
                    weight_decay=self.arg.weight_decay
                )
            except:
                self.print_log("AdamW not available, using Adam with weight decay")
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
        elif self.arg.optimizer.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.arg.base_lr)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
    
    def load_data(self):
        if self.arg.phase == 'train':
            builder = self.prepare_smartfallmm_tf(self.arg)
            use_dtw = self.arg.dataset_args.get('use_dtw', True)
            verbose = self.arg.dataset_args.get('verbose', True)
            
            self.norm_train = self.split_by_subjects_tf(builder, self.train_subjects, self.fuse, use_dtw, verbose)
            self.norm_val = self.split_by_subjects_tf(builder, self.val_subject, self.fuse, use_dtw, verbose)
            if self.has_empty_value(list(self.norm_val.values())):
                return False
                
            self.data_loader['train'] = self.UTD_MM_TF(self.norm_train, self.arg.batch_size)
            self.data_loader['val'] = self.UTD_MM_TF(self.norm_val, self.arg.val_batch_size)
            
            self.cal_weights()
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            self.norm_test = self.split_by_subjects_tf(builder, self.test_subject, self.fuse, use_dtw, verbose)
            if self.has_empty_value(list(self.norm_test.values())):
                return False
                
            self.data_loader['test'] = self.UTD_MM_TF(self.norm_test, self.arg.test_batch_size)
            self.distribution_viz(self.norm_test['labels'], self.arg.work_dir, f'test_{self.test_subject[0]}')
            
            return True
    
    def distribution_viz(self, labels, work_dir, mode):
        values, count = np.unique(labels, return_counts=True)
        plt.figure()
        plt.bar(values, count)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title(f'{mode.capitalize()} Label Distribution')
        plt.savefig(f'{work_dir}/{mode}_Label_Distribution.png')
        plt.close()
    
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    def print_log(self, string, print_time=True):
        print(string)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(string, file=f)
    
    def loss_viz(self, train_loss, val_loss):
        if not train_loss or not val_loss or len(train_loss) == 0 or len(val_loss) == 0:
            return
            
        epochs = range(len(train_loss))
        plt.figure()
        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title(f'Train vs Val Loss for {self.test_subject[0]}')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'{self.arg.work_dir}/trainvsval_{self.test_subject[0]}.png')
        plt.close()
    
    def create_df(self):
        return []
    
    def save_model(self, model, save_path, save_format='weights'):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if save_format == 'weights':
                model.save_weights(save_path, save_format='h5')
                self.print_log(f"Model weights saved to {save_path}")
            elif save_format == 'keras':
                if save_path.endswith('.h5'):
                    save_path = save_path[:-3] + '.keras'
                model.save(save_path, save_format='keras')
                self.print_log(f"Full model saved to {save_path}")
            else:
                model.save(save_path, save_format='h5')
                self.print_log(f"Full model saved to {save_path} in HDF5 format")
                
            return True
        except Exception as e:
            self.print_log(f"Error saving model: {str(e)}")
            self.print_log(traceback.format_exc())
            return False

    def export_to_tflite(self, model, output_path, sample_batch=None):
        try:
            if sample_batch is None:
                sample_shape = (1, 128, 4)
                sample_input = tf.constant(np.zeros(sample_shape, dtype=np.float32))
                sample_dict = {'accelerometer': sample_input}
            else:
                sample_dict = sample_batch
            
            # Create a wrapped model with a defined signature
            class WrappedModel(tf.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                @tf.function(input_signature=[tf.TensorSpec(shape=[1, 128, 4], dtype=tf.float32)])
                def __call__(self, x):
                    logits, _ = self.model(x, training=False)
                    return {'output': tf.sigmoid(logits)}
                    
            wrapped_model = WrappedModel(model)
            
            # Test the wrapped model
            wrapped_model(sample_input)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [wrapped_model.__call__.get_concrete_function()]
            )
            
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
                
            self.print_log(f"Successfully exported model to TFLite: {output_path}")
            return True
        except Exception as e:
            self.print_log(f"Error exporting to TFLite: {str(e)}")
            self.print_log(traceback.format_exc())
            return False
    
    def train(self, epoch):
        self.model.trainable = True
        self.record_time()
        
        loader = self.data_loader['train']
        timer = {'dataloader': 0.001, 'model': 0.001, 'stats': 0.001}
        
        label_list = []
        pred_list = []
        train_loss = 0
        cnt = 0
        
        # Initialize optimizer variables with a warm-up step if needed
        try:
            # Warm-up step to initialize optimizer variables
            if epoch == 0:
                warm_data = next(iter(loader))[0]
                with tf.GradientTape() as tape:
                    logits, _ = self.model(warm_data, training=True)
                    dummy_loss = tf.reduce_mean(logits)
                gradients = tape.gradient(dummy_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        except Exception as e:
            self.print_log(f"Warm-up step failed, but training will continue: {e}")
        
        # Main training loop - WITHOUT tf.function to avoid variable creation issues
        for batch_idx in tqdm(range(len(loader)), desc=f"Epoch {epoch+1}"):
            try:
                batch_data, batch_targets, _ = loader[batch_idx]
                timer['dataloader'] += self.split_time()
                
                # Manual training step
                with tf.GradientTape() as tape:
                    logits, _ = self.model(batch_data, training=True)
                    logits = tf.reshape(logits, [-1])
                    batch_targets = tf.cast(batch_targets, tf.float32)
                    loss = self.criterion(batch_targets, logits)
                    loss = tf.reduce_mean(loss)
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                train_loss += loss.numpy()
                preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                label_list.extend(batch_targets.numpy())
                pred_list.extend(preds.numpy())
                cnt += 1
                
                timer['model'] += self.split_time()
                timer['stats'] += self.split_time()
            except Exception as e:
                self.print_log(f"Error in batch {batch_idx}: {e}")
                self.print_log(traceback.format_exc())
        
        if cnt > 0:
            train_loss /= cnt
            metrics_dict, (accuracy, f1, recall, precision, auc_score) = self.calculate_metrics(
                label_list, pred_list
            )
            
            self.train_loss_summary.append(train_loss)
            
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
                for k, v in timer.items()
            }
            
            self.print_log(
                f'\tTraining Loss: {train_loss:.4f}, Acc: {accuracy:.2f}%, F1: {f1:.2f}%, '
                f'Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%'
            )
            
            self.print_log(f'\tTime consumption: [Data]{proportion["dataloader"]}, [Network]{proportion["model"]}')
            
            val_loss, val_metrics = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
            self.val_loss_summary.append(val_loss)
    
    def eval(self, epoch, loader_name='val', result_file=None):
        self.model.trainable = False
        
        if result_file is not None:
            f_r = open(result_file, 'w')
        
        self.print_log(f'Eval epoch: {epoch+1}')
        
        loss = 0
        cnt = 0
        label_list = []
        pred_list = []
        
        loader = self.data_loader[loader_name]
        
        for batch_idx in tqdm(range(len(loader)), desc=f"Evaluating {loader_name}"):
            try:
                batch_data, batch_targets, _ = loader[batch_idx]
                
                # Forward pass without tf.function
                logits, _ = self.model(batch_data, training=False)
                logits = tf.reshape(logits, [-1])
                batch_targets = tf.cast(batch_targets, tf.float32)
                batch_loss = self.criterion(batch_targets, logits)
                batch_loss = tf.reduce_mean(batch_loss)
                
                loss += batch_loss.numpy()
                preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
                label_list.extend(batch_targets.numpy())
                pred_list.extend(preds.numpy())
                cnt += 1
            except Exception as e:
                self.print_log(f"Error in evaluation batch {batch_idx}: {e}")
                self.print_log(traceback.format_exc())
        
        if cnt > 0:
            loss = loss / cnt
            metrics_dict, (accuracy, f1, recall, precision, auc_score) = self.calculate_metrics(
                label_list, pred_list
            )
            
            if result_file is not None:
                for i, x in enumerate(pred_list):
                    f_r.write(f"{x} ==> {label_list[i]}\n")
                f_r.close()
            
            self.print_log(
                f'{loader_name.capitalize()} Loss: {loss:.4f}, Acc: {accuracy:.2f}%, '
                f'F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, AUC: {auc_score:.2f}%'
            )
            
            if loader_name == 'val':
                is_best_loss = loss < self.best_loss
                is_best_f1 = f1 > self.best_f1
                
                if is_best_loss:
                    self.best_loss = loss
                    self.print_log(f"New best validation loss: {loss:.6f}")
                
                if is_best_f1:
                    self.best_f1 = f1
                    self.print_log(f"New best validation F1: {f1:.2f}%")
                
                if is_best_loss or is_best_f1:
                    checkpoint_dir = os.path.join(self.arg.work_dir, 'models')
                    model_save_path = os.path.join(checkpoint_dir, f'{self.arg.model_saved_name}_{self.test_subject[0]}.h5')
                    
                    # Save weights only - more reliable
                    if self.save_model(self.model, model_save_path, save_format='weights'):
                        # Only try to export to TFLite once per fold with the best model
                        if not hasattr(self, f'exported_tflite_{self.test_subject[0]}'):
                            tflite_path = os.path.join(checkpoint_dir, f'{self.arg.model_saved_name}_{self.test_subject[0]}.tflite')
                            if self.export_to_tflite(self.model, tflite_path, batch_data):
                                setattr(self, f'exported_tflite_{self.test_subject[0]}', True)
                
                self.early_stop(loss)
            else:
                self.test_accuracy = accuracy
                self.test_f1 = f1
                self.test_recall = recall
                self.test_precision = precision
                self.test_auc = auc_score
            
            return loss, metrics_dict
        
        return float('inf'), None
    
    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')
            
            results = []
            
            val_subjects = [38, 46]
            
            for test_subject in self.arg.subjects:
                if test_subject in val_subjects:
                    continue
                    
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.best_f1 = 0.0
                
                train_subjects = [s for s in self.arg.subjects if s != test_subject and s not in val_subjects]
                self.val_subject = val_subjects
                self.test_subject = [test_subject]
                self.train_subjects = train_subjects
                
                self.print_log(f"=== Testing on subject {test_subject} ===")
                self.print_log(f"Train subjects: {self.train_subjects}")
                self.print_log(f"Val subjects: {self.val_subject}")
                
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
                
                if not self.load_data():
                    self.print_log(f"Skipping subject {test_subject} due to data loading failure")
                    continue
                
                try:
                    self.load_optimizer()
                    self.load_loss()
                    self.early_stop.early_stop = False  # Reset early stopping
                    
                    for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                        self.train(epoch)
                        if self.early_stop.early_stop:
                            self.print_log("Early stopping triggered")
                            break
                    
                    # Load best model for testing
                    best_model_path = os.path.join(self.arg.work_dir, 'models', f'{self.arg.model_saved_name}_{self.test_subject[0]}.h5')
                    if os.path.exists(best_model_path):
                        try:
                            self.model = self.load_model(self.arg.model, self.arg.model_args)
                            self.model.load_weights(best_model_path)
                            self.print_log(f"Loaded weights from {best_model_path}")
                        except Exception as e:
                            self.print_log(f"Error loading saved model weights: {e}")
                    
                    self.print_log(f'Testing subject {self.test_subject[0]}')
                    _, test_metrics = self.eval(epoch=0, loader_name='test')
                    
                    if len(self.train_loss_summary) > 0 and len(self.val_loss_summary) > 0:
                        self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                    
                    results.append({
                        'test_subject': str(self.test_subject[0]),
                        'accuracy': round(self.test_accuracy, 2),
                        'f1_score': round(self.test_f1, 2),
                        'precision': round(self.test_precision, 2),
                        'recall': round(self.test_recall, 2),
                        'auc': round(self.test_auc, 2)
                    })
                except Exception as e:
                    self.print_log(f"Error during training for subject {test_subject}: {e}")
                    self.print_log(traceback.format_exc())
            
            if results:
                avg_result = {'test_subject': 'Average'}
                for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
                    avg_result[metric] = round(
                        sum(float(r[metric]) for r in results) / len(results), 
                        2
                    )
                results.append(avg_result)
                
                with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                    json.dump(results, f, indent=4)
                    
                self.print_log(f"Results saved to {self.arg.work_dir}/scores.json")
