import os
import time
import datetime
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, arg):
        from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf, UTD_MM_TF
        from utils.callbacks import EarlyStoppingTF, CheckpointManagerTF
        from utils.metrics import calculate_metrics
        
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
        
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.arg.work_dir)
        
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'
        self.save_config(arg.config, arg.work_dir)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.arg.device)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
        
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.model = tf.keras.models.load_model(self.arg.weights)
        
        self.include_val = arg.include_val
        self.calculate_metrics = calculate_metrics
        self.prepare_smartfallmm_tf = prepare_smartfallmm_tf
        self.split_by_subjects_tf = split_by_subjects_tf
        self.UTD_MM_TF = UTD_MM_TF
        self.CheckpointManagerTF = CheckpointManagerTF
        
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size: {num_params/(1024 ** 2):.2f} MB')
    
    def add_avg_df(self, results):
        avg_dict = {}
        for col in results[0].keys():
            if col != 'test_subject':
                vals = [float(r[col]) for r in results]
                avg_dict[col] = round(sum(vals) / len(vals), 2)
            else:
                avg_dict[col] = 'Average'
        results.append(avg_dict)
        return results
    
    def save_config(self, src_path, dest_path):
        config_filename = os.path.basename(src_path)
        with open(src_path, 'r') as f_src:
            with open(f'{dest_path}/{config_filename}', 'w') as f_dst:
                f_dst.write(f_src.read())
    
    def cal_weights(self):
        labels = self.norm_train['labels']
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        self.pos_weights = num_neg / num_pos if num_pos > 0 else 1.0
    
    def count_parameters(self, model):
        try:
            dummy_input = {'accelerometer': tf.zeros((1, 128, 4))}
            _ = model(dummy_input, training=False)
            
            total_params = 0
            for variable in model.trainable_variables:
                total_params += np.prod(variable.shape)
                
            if total_params == 0:
                total_params = sum(np.prod(v.get_shape()) for v in model.trainable_weights)
                
            return total_params
        except:
            return 0
    
    def has_empty_value(self, *lists):
        return any(len(lst) == 0 for lst in lists)
    
    def load_model(self, model_name, model_args):
        from utils.common import import_class
        Model = import_class(model_name)
        model = Model(**model_args)
        return model
    
    def load_loss(self):
        if self.arg.loss.lower() == 'bce':
            self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif self.arg.loss.lower() == 'binary_focal':
            class BinaryFocalLoss(tf.keras.losses.Loss):
                def __init__(self, alpha=0.75, gamma=2.0, from_logits=True):
                    super(BinaryFocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.from_logits = from_logits
                
                def call(self, y_true, y_pred):
                    y_true = tf.cast(y_true, tf.float32)
                    if self.from_logits:
                        prob = tf.sigmoid(y_pred)
                    else:
                        prob = y_pred
                    pt = tf.where(tf.equal(y_true, 1.0), prob, 1 - prob)
                    alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
                    focal_loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
                    return tf.reduce_mean(focal_loss)
            
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
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
        elif self.arg.optimizer.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.arg.base_lr)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
    
    def load_data(self):
        if self.arg.phase == 'train':
            builder = self.prepare_smartfallmm_tf(self.arg)
            
            self.norm_train = self.split_by_subjects_tf(builder, self.train_subjects, self.fuse)
            self.norm_val = self.split_by_subjects_tf(builder, self.val_subject, self.fuse)
            
            if self.has_empty_value(list(self.norm_val.values())):
                return False
            
            self.data_loader['train'] = self.UTD_MM_TF(self.norm_train, self.arg.batch_size)
            self.data_loader['val'] = self.UTD_MM_TF(self.norm_val, self.arg.val_batch_size)
            
            self.cal_weights()
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            self.norm_test = self.split_by_subjects_tf(builder, self.test_subject, self.fuse)
            
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
    
    @tf.function
    def train_step(self, data, labels):
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
        logits, _ = self.model(data, training=False)
        if len(logits.shape) > 1:
            logits = tf.squeeze(logits, axis=-1)
        loss = self.criterion(labels, logits)
        preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.int32)
        return loss, preds
    
    def train(self, epoch):
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
        
        train_loss /= cnt
        accuracy, f1, recall, precision, auc_score = self.calculate_metrics(label_list, pred_list)
        
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
            accuracy, f1, recall, precision, auc_score = self.calculate_metrics(label_list, pred_list)
            
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
                    ckpt_manager = self.CheckpointManagerTF(
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
        if self.arg.phase == 'train':
            # Print parameters without using json.dumps to avoid serialization issues
            self.print_log('Parameters:')
            for key, value in vars(self.arg).items():
                self.print_log(f'  {key}: {value}')
            self.print_log('')
            
            results = self.create_df()
            
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
                    
                    ckpt_manager = self.CheckpointManagerTF(
                        self.model, 
                        self.optimizer, 
                        f'{self.model_path}_{self.test_subject[0]}'
                    )
                    ckpt_manager.load_best_checkpoint()
                    
                    self.print_log(f' ------------ Test Subject {self.test_subject[0]} -------')
                    self.eval(epoch=0, loader_name='test')
                    self.print_log(f'Test accuracy: {self.test_accuracy}')
                    self.print_log(f'Test F-Score: {self.test_f1}')
                    
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
                    self.print_log(f"Error during training/testing for subject {self.test_subject[0]}: {e}")
                    import traceback
                    self.print_log(traceback.format_exc())
            
            if results:
                results = self.add_avg_df(results)
                import json
                
                # Convert any problematic types to strings first
                for i, result in enumerate(results):
                    results[i] = {k: str(v) if not isinstance(v, (int, float, str)) else v 
                                  for k, v in result.items()}
                
                with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                    json.dump(results, f, indent=4)
            else:
                self.print_log("No results to save.")
