import os
import argparse
import yaml
import tensorflow as tf

from utils.common import str2bool, init_seed
from trainer.base_trainer import Trainer

def get_args():
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
    parser.add_argument('--mixed-precision', type=str2bool, default=False)
    return parser

if __name__ == "__main__":
    # Parse arguments
    parser = get_args()
    args = parser.parse_args()
    
    # Load configuration from YAML
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update arguments from config
        for k, v in config.items():
            if k not in vars(args):
                print(f'WARNING: Config key "{k}" not in arguments')
            setattr(args, k, v)
    
    # Initialize random seed
    init_seed(args.seed)
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            print(e)
    
    # Create trainer and start training/testing
    trainer = Trainer(args)
    trainer.start()
