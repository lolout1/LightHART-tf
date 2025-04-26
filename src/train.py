#!/usr/bin/env python
import os
import sys
import time
import datetime
import argparse
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Configure environment for GPU in WSL
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/usr/lib/wsl/lib'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
from utils.callbacks_tf import EarlyStoppingTF

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Unsupported value encountered.')

def init_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    try:
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)
    except (ImportError, AttributeError):
        if mod_str.startswith('src.'):
            alt_mod_str = mod_str[4:]
            try:
                __import__(alt_mod_str)
                return getattr(sys.modules[alt_mod_str], class_str)
            except: pass
        raise ImportError(f"Cannot import {import_str}")

class Trainer:
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
        
        if os.path.exists(self.arg.work_dir):
            self.arg.work_dir = f"{self.arg.work_dir}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.arg.work_dir)
        
        self.model_path = f'{self.arg.work_dir}/{self.arg.model_saved_name}'
        self.save_config(arg.config, arg.work_dir)
        
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
        
        if self.arg.phase == 'train':
            self.model = self.load_model(arg.model, arg.model_args)
        else:
            self.model = tf.keras.models.load_model(self.arg.weights)
        
        self.include_val = arg.include_val
        num_params = self.count_parameters(self.model)
        self.print_log(f'# Parameters: {num_params}')
        self.print_log(f'Model size : {num_params/(1024 ** 2):.2f} MB')
    
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
    
    def save_config(self, src_path, desc_path):
        with open(src_path, 'r') as f_src:
            with open(f'{desc_path}/{os.path.basename(src_path)}', 'w') as f_dst:
                f_dst.write(f_src.read())
    
    def cal_weights(self):
        labels = self.norm_train['labels']
        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)
        self.pos_weights = num_neg / num_pos if num_pos > 0 else 1.0
    
    def count_parameters(self, model):
        return np.sum([np.prod(v.get_shape()) for v in model.trainable_variables])
    
    def has_empty_value(self, *lists):
        return any(len(lst) == 0 for lst in lists)
    
    def load_model(self, model_name, model_args):
        Model = import_class(model_name)
        model = Model(**model_args)
        return model
    
    def load_loss(self):
        if self.arg.loss.lower() == 'bce':
            self.criterion = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                pos_weight=tf.constant(self.pos_weights)
            )
        elif self.arg.loss.lower() == 'binary_focal':
            self.criterion = BinaryFocalLossTF(alpha=0.75)
        else:
            self.criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def load_optimizer(self):
        if self.arg.optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.arg.base_lr)
        elif self.arg.optimizer.lower() == 'adamw':
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.arg.base_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.arg.optimizer}")
    
    def load_data(self):
        if self.arg.phase == 'train':
            builder = prepare_smartfallmm_tf(self.arg)
            
            self.norm_train = split_by_subjects_tf(builder, self.train_subjects, self.fuse)
            self.norm_val = split_by_subjects_tf(builder, self.val_subject, self.fuse)
            
            if self.has_empty_value(list(self.norm_val.values())):
                return False
            
            # Create data loaders (replace with your actual data loader initialization)
            Feeder = import_class(self.arg.feeder)
            self.data_loader['train'] = Feeder(self.norm_train, self.arg.batch_size)
            self.data_loader['val'] = Feeder(self.norm_val, self.arg.val_batch_size)
            
            self.cal_weights()
            self.distribution_viz(self.norm_train['labels'], self.arg.work_dir, 'train')
            
            self.norm_test = split_by_subjects_tf(builder, self.test_subject, self.fuse)
            
            if self.has_empty_value(list(self.norm_test.values())):
                return False
                
            self.data_loader['test'] = Feeder(self.norm_test, self.arg.test_batch_size)
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
    
    def cm_viz(self, y_pred, y_true):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(np.unique(y_true))
        plt.yticks(np.unique(y_true))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.savefig(f'{self.arg.work_dir}/Confusion_Matrix.png')
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
            loss, preds = self.train_step(inputs, targets)
            
            train_loss += loss
            label_list.extend(targets.numpy())
            pred_list.extend(preds.numpy())
            cnt += 1
            
            timer['model'] += self.split_time()
            timer['stats'] += self.split_time()
        
        train_loss /= cnt
        accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)
        
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
    
    def cal_metrics(self, targets, predictions):
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        f1 = f1_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        try:
            auc_score = roc_auc_score(targets, predictions)
        except:
            auc_score = 0.5
            
        accuracy = accuracy_score(targets, predictions)
        
        return accuracy*100, f1*100, recall*100, precision*100, auc_score*100
    
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
            targets = tf.cast(targets, tf.int32)
            batch_loss, preds = self.test_step(inputs, targets)
            
            loss += batch_loss
            label_list.extend(targets.numpy())
            pred_list.extend(preds.numpy())
            cnt += 1
        
        loss /= cnt
        accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)
        
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
                self.model.save(f'{self.model_path}_{self.test_subject[0]}')
                self.print_log('Weights Saved')
        else:
            self.test_accuracy = accuracy
            self.test_f1 = f1
            self.test_recall = recall
            self.test_precision = precision
            self.test_auc = auc_score
        
        return loss.numpy()
    
    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters: \n{yaml.dump(vars(self.arg), default_flow_style=False)}\n')
            
            results = self.create_df()
            
            for i in range(len(self.arg.subjects[:-3])):
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                
                test_subject = self.arg.subjects[i]
                train_subjects = list(filter(lambda x: x != test_subject, self.arg.subjects))
                self.val_subject = [38, 46]
                self.test_subject = [test_subject]
                self.train_subjects = train_subjects
                
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                self.print_log(f'Model Parameters: {self.count_parameters(self.model)}')
                
                if not self.load_data():
                    continue
                
                self.load_optimizer()
                self.load_loss()
                
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                    if self.early_stop.early_stop:
                        self.early_stop = EarlyStoppingTF(patience=15, min_delta=.001)
                        break
                
                self.model = tf.keras.models.load_model(f'{self.model_path}_{self.test_subject[0]}')
                
                self.print_log(f'------------ Test Subject {self.test_subject[0]} -------')
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
            
            results = self.add_avg_df(results)
            
            with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fall Detection Training')
    parser.add_argument('--config', default='./config/smartfallmm/teacher.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--num-epoch', type=int, default=70)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--model', default=None, help='Name of model to load')
    parser.add_argument('--device', default='0')
    parser.add_argument('--model-args', default=str, help='Model arguments')
    parser.add_argument('--weights', type=str, help='Location of weight file')
    parser.add_argument('--model-saved-name', type=str, default='test')
    parser.add_argument('--loss', default='bce', help='Name of loss function')
    parser.add_argument('--loss-args', default="{}", type=str)
    parser.add_argument('--dataset-args', default=str)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--feeder', default=None)
    parser.add_argument('--train-feeder-args', default=str)
    parser.add_argument('--val-feeder-args', default=str)
    parser.add_argument('--test_feeder_args', default=str)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--work-dir', type=str, default='experiments/student')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        for k, v in config.items():
            if k not in vars(args):
                print(f'WRONG ARG: {k}')
                continue
            setattr(args, k, v)
    
    init_seed(args.seed)
    trainer = Trainer(args)
    trainer.start()
