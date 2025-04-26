import os
import time
import datetime
import argparse
import yaml
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Import custom modules
from utils.dataset_tf import prepare_smartfallmm_tf, split_by_subjects_tf
from utils.callbacks_tf import EarlyStoppingTF, CheckpointManagerTF
from utils.loss_tf import BinaryFocalLossTF, DistillationLossTF
from models.transformer_tf import TransModelTF
from feeder.make_dataset_tf import UTD_MM_TF
from train import Trainer, get_args, str2bool, init_seed, import_class

class DistillerTF(Trainer):
    def __init__(self, arg):
        super().__init__(arg)
        # Load teacher model
        self.teacher_model = self.load_model(arg.teacher_model, arg.teacher_args)
        self.early_stop = EarlyStoppingTF(patience=15, min_delta=.001)
        
        # Initialize distillation parameters
        self.temperature = 4.0
        self.alpha = 0.5
    
    def load_teacher_weights(self):
        '''Load teacher model weights'''
        # Load weights to the teacher model
        checkpoint = tf.train.Checkpoint(model=self.teacher_model)
        status = checkpoint.restore(tf.train.latest_checkpoint(
            f"{self.arg.teacher_weight}_{self.test_subject[0]}"
        ))
        status.expect_partial()
        
        # Set teacher model to non-trainable
        self.teacher_model.trainable = False
    
    def load_loss(self):
        '''Load distillation loss function'''
        self.criterion = DistillationLossTF(
            temperature=self.temperature,
            alpha=self.alpha
        )
    
    @tf.function
    def train_step(self, data, labels):
        '''Single training step with distillation'''
        # Teacher forward pass (no gradients)
        teacher_logits, teacher_features = self.teacher_model(data, training=False)
        
        with tf.GradientTape() as tape:
            # Student forward pass
            student_logits, student_features = self.model(data, training=True)
            
            # Compute correct predictions from teacher to weight distillation
            teacher_preds = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
            correct_mask = tf.cast(tf.equal(teacher_preds, tf.cast(labels, tf.float32)), tf.float32)
            weights = (1.0 / 1.5) * correct_mask + (0.5 / 1.5) * (1 - correct_mask)
            
            # Compute distillation loss
            loss = self.criterion((
                student_logits, 
                teacher_logits, 
                labels, 
                student_features, 
                teacher_features, 
                weights
            ))
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate predictions from student
        preds = tf.cast(tf.sigmoid(student_logits) > 0.5, tf.int32)
        
        return loss, preds
    
    def visualize_features(self, teacher_features, student_features, epoch):
        '''Visualize feature distributions between teacher and student'''
        # Flatten features for visualization
        teacher_flat = tf.reshape(teacher_features, [teacher_features.shape[0], -1])
        student_flat = tf.reshape(student_features, [student_features.shape[0], -1])
        
        # Create histograms for first few samples
        plt.figure(figsize=(12, 6))
        for i in range(min(8, teacher_flat.shape[0])):
            plt.subplot(2, 4, i+1)
            
            # Plot teacher feature distribution
            plt.hist(
                teacher_flat[i].numpy(), 
                bins=30, 
                alpha=0.5, 
                label='Teacher', 
                color='blue'
            )
            
            # Plot student feature distribution
            plt.hist(
                student_flat[i].numpy(), 
                bins=30, 
                alpha=0.5, 
                label='Student', 
                color='red'
            )
            
            plt.legend()
            plt.title(f'Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig(f'{self.arg.work_dir}/Feature_KDE_{epoch}.png')
        plt.close()
    
    def train(self, epoch):
        '''Train model for one epoch with distillation'''
        self.model.trainable = True
        self.teacher_model.trainable = False
        self.record_time()
        
        loader = self.data_loader['train']
        timer = {'dataloader': 0.001, 'model': 0.001, 'stats': 0.001}
        
        label_list = []
        pred_list = []
        train_loss = 0
        cnt = 0
        
        for batch_idx, (inputs, targets, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            timer['dataloader'] += self.split_time()
            
            # Convert targets to float32 for loss calculation
            targets = tf.cast(targets, tf.float32)
            
            # Train step with distillation
            loss, preds = self.train_step(inputs, targets)
            
            # Update stats
            train_loss += loss
            label_list.extend(tf.cast(targets, tf.int32).numpy())
            pred_list.extend(preds.numpy())
            cnt += 1
            
            timer['model'] += self.split_time()
            
            # Visualize feature distributions periodically
            if epoch % 10 == 0 and batch_idx == 0:
                # Get features for visualization
                _, teacher_features = self.teacher_model(inputs, training=False)
                _, student_features = self.model(inputs, training=False)
                self.visualize_features(teacher_features, student_features, epoch)
            
            timer['stats'] += self.split_time()
        
        # Calculate metrics
        train_loss /= cnt
        accuracy, f1, recall, precision, auc_score = self.cal_metrics(label_list, pred_list)
        
        self.train_loss_summary.append(train_loss.numpy())
        
        # Log results
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        
        self.print_log(
            '\tTraining Loss: {:4f},  Acc: {:2f}%, F1 score: {:2f}%, '
            'Precision: {:2f}%, Recall: {:2f}%, AUC: {:2f}%'.format(
                train_loss.numpy(), accuracy, f1, precision, recall, auc_score
            )
        )
        
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        
        # Validate
        val_loss = self.eval(epoch, loader_name='val', result_file=self.arg.result_file)
        self.val_loss_summary.append(val_loss)
        
        # Check early stopping
        self.early_stop(val_loss)
    
    def start(self):
        '''Start training or testing with distillation'''
        if self.arg.phase == 'train':
            self.print_log(f'Parameters: \n{json.dumps(vars(self.arg), indent=4)}\n')
            
            results = self.create_df()
            
            # Loop through test subjects for cross-validation
            for i in range(len(self.arg.subjects[:-3])):
                # Reset metrics for this fold
                self.train_loss_summary = []
                self.val_loss_summary = []
                self.best_loss = float('inf')
                self.best_f1 = float('-inf')
                self.best_recall = float('-inf')
                self.best_precision = float('-inf')
                self.best_accuracy = float('-inf')
                
                # Define subject splits
                test_subject = self.arg.subjects[i]
                train_subjects = list(filter(lambda x: x != test_subject, self.arg.subjects))
                self.val_subject = [35, 46]
                self.test_subject = [test_subject]
                self.train_subjects = train_subjects
                
                # Load teacher model weights
                self.load_teacher_weights()
                
                # Create a new student model for this fold
                self.model = self.load_model(self.arg.model, self.arg.model_args)
                
                # Load data
                if not self.load_data():
                    continue
                
                # Setup optimizer and loss
                self.load_optimizer()
                self.load_loss()
                
                # Train for specified epochs
                for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                    self.train(epoch)
                    if self.early_stop.early_stop:
                        self.early_stop = EarlyStoppingTF(patience=20, min_delta=1e-6)
                        break
                
                # Load best model for testing
                ckpt_manager = CheckpointManagerTF(
                    self.model, 
                    self.optimizer, 
                    f'{self.model_path}_{self.test_subject[0]}'
                )
                ckpt_manager.load_best_checkpoint()
                
                # Evaluate on test set
                self.print_log(f' ------------ Test Subject {self.test_subject[0]} -------')
                self.eval(epoch=0, loader_name='test')
                self.print_log(f'Test accuracy: {self.test_accuracy}')
                self.print_log(f'Test F-Score: {self.test_f1}')
                
                # Visualize loss curves
                self.loss_viz(self.train_loss_summary, self.val_loss_summary)
                
                # Record results
                results.append({
                    'test_subject': str(self.test_subject[0]),
                    'accuracy': round(self.test_accuracy, 2),
                    'f1_score': round(self.test_f1, 2),
                    'precision': round(self.test_precision, 2),
                    'recall': round(self.test_recall, 2),
                    'auc': round(self.test_auc, 2)
                })
            
            # Add average row to results
            results = self.add_avg_df(results)
            
            # Save results to file
            with open(f'{self.arg.work_dir}/scores.json', 'w') as f:
                json.dump(results, f, indent=4)

def get_distill_args():
    '''Get arguments for distillation'''
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--config', default='./config/smartfallmm/distill.yaml')
    parser.add_argument('--dataset', type=str, default='utd')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=8)
    parser.add_argument('--num-epoch', type=int, default=70)
    parser.add_argument('--start-epoch', type=int, default=0)
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--base-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    
    # Data parameters
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--dataset-args', default=None, type=str)
    
    # Teacher model parameters
    parser.add_argument('--teacher-model', default=None, help='Name of teacher model')
    parser.add_argument('--teacher-args', default=str, help='Teacher model arguments')
    parser.add_argument('--teacher-weight', type=str, help='Path to teacher model weights')
    
    # Student model parameters
    parser.add_argument('--model', default=None, help='Name of student model')
    parser.add_argument('--model-args', default=str, help='Student model arguments')
    parser.add_argument('--device', default='0')
    parser.add_argument('--model-saved-name', type=str, default='student')
    
    # Distillation parameters
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    
    # Other parameters
    parser.add_argument('--feeder', default=None)
    parser.add_argument('--train-feeder-args', default=str)
    parser.add_argument('--val-feeder-args', default=str)
    parser.add_argument('--test_feeder_args', default=str)
    parser.add_argument('--include-val', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--work-dir', type=str, default='exps/test')
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-worker', type=int, default=0)
    parser.add_argument('--result-file', type=str)
    
    # Mixed precision
    parser.add_argument('--mixed-precision', type=str2bool, default=False)
    
    return parser

if __name__ == "__main__":
    parser = get_distill_args()
    
    # Load arguments from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WRONG ARG: {k}')
                assert k in key
        
        parser.set_defaults(**default_arg)
    
    arg = parser.parse_args()
    
    # Set random seed
    init_seed(arg.seed)
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Start distillation
    distiller = DistillerTF(arg)
    distiller.start()
