import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from functools import partial
from sklearn.preprocessing import StandardScaler
from utils.processor_tf import butterworth_filter, pad_sequence_tf, align_sequence_dtw, selective_sliding_window, sliding_window

logger = logging.getLogger(__name__)


def csvloader(file_path, **kwargs):
    """
    Enhanced CSV loader with robust 30Hz interpolation and timestamp validation
    """
    try:
        sampling_method = kwargs.get('sampling', 'none')
        target_fs = kwargs.get('target_fs', 30)  # Fixed at 30Hz
        viz_dir = kwargs.get('viz_dir', None)
        trial_name = kwargs.get('trial_name', None)
        
        if 'accelerometer' in file_path and sampling_method == 'hybrid':
            logger.debug(f"Using robust 30Hz interpolation for {file_path}")
            
            # Import here to avoid circular imports
            try:
                from utils.hybrid_interpolation import RobustInterpolator
            except ImportError as e:
                logger.error(f"Cannot import RobustInterpolator: {e}")
                return None
            
            # Quick timestamp validation
            try:
                with open(file_path, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(3)]
                
                has_timestamps = False
                for line in first_lines:
                    if line and ('/' in line or '-' in line or ':' in line):
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                pd.to_datetime(parts[0])
                                has_timestamps = True
                                break
                            except:
                                continue
                
                if not has_timestamps:
                    logger.warning(f"No valid timestamps in {file_path}, skipping")
                    return None
                    
            except Exception as e:
                logger.error(f"Error checking timestamps in {file_path}: {e}")
                return None
            
            # Extract sampling arguments
            sampling_args = {
                'max_upsample_ratio': kwargs.get('max_upsample_ratio', 1.5),
                'min_samples': kwargs.get('min_samples', 10),
                'gap_threshold_seconds': kwargs.get('gap_threshold_seconds', 0.5)
            }
            
            # Process with robust interpolator
            try:
                # Load CSV
                df = pd.read_csv(file_path, header=None, names=['timestamp', 'x', 'y', 'z'], 
                               on_bad_lines='skip')
                
                # Basic validation
                df = df.dropna()
                for col in ['x', 'y', 'z']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['x', 'y', 'z'])
                
                if len(df) < sampling_args['min_samples']:
                    logger.warning(f"Too few samples in {file_path}: {len(df)}")
                    return None

                # Create interpolator
                interpolator = RobustInterpolator(
                    target_fs=target_fs, 
                    viz_dir=viz_dir,
                    **sampling_args
                )
                
                # Validate timestamps
                result = interpolator.validate_and_parse_timestamps(df['timestamp'])
                if result is None:
                    logger.warning(f"Invalid timestamps in {file_path}, skipping")
                    return None
                
                t_seconds, valid_mask = result
                acc_data = df[['x', 'y', 'z']].values[valid_mask].astype(np.float32)
                
                # Analyze and interpolate
                stats = interpolator.analyze_sampling_characteristics(t_seconds)
                if stats is None:
                    logger.warning(f"Cannot analyze sampling for {file_path}")
                    return None
                
                t_uniform, data_resampled, method_used = interpolator.interpolate_to_30hz(
                    t_seconds, acc_data, stats
                )
                
                if data_resampled is None:
                    logger.warning(f"Interpolation failed for {file_path}")
                    return None

                # Create visualization if requested
                if viz_dir and trial_name:
                    interpolator.visualize_comparison(
                        t_seconds, acc_data, t_uniform, data_resampled, stats, method_used, trial_name
                    )

                # Log success
                duration = t_seconds[-1] - t_seconds[0] if len(t_seconds) > 1 else 0
                achieved_fs = (len(data_resampled) - 1) / duration if duration > 0 else 0
                
                logger.info(f"30Hz resampled {trial_name}: {len(acc_data)} -> {len(data_resampled)} samples "
                           f"(Achieved: {achieved_fs:.1f}Hz, Method: {method_used})")
                
                return data_resampled
                
            except Exception as e:
                logger.error(f"Error processing {file_path} with 30Hz interpolation: {e}")
                return None
                
        else:
            # Standard CSV loading for non-accelerometer data
            try:
                file_data = pd.read_csv(file_path, index_col=False, header=0).dropna().bfill()
                num_col = file_data.shape[1]
                cols = 96 if 'skeleton' in file_path else 3
                activity_data = file_data.iloc[2:, -cols:].to_numpy(dtype=np.float32)
                activity_data = activity_data[~np.isnan(activity_data).any(axis=1)]
                
                if activity_data.shape[0] < 10:
                    logger.warning(f"Too few samples in {file_path}: {activity_data.shape}")
                    return None
                    
                return activity_data
            except Exception as e:
                logger.error(f"Error loading standard CSV {file_path}: {e}")
                return None
            
    except Exception as e:
        logger.error(f"Error in csvloader for {file_path}: {e}")
        return None


def matloader(file_path, **kwargs):
    """MATLAB file loader"""
    try:
        from scipy.io import loadmat
        key = kwargs.get('key', None)
        if key not in ['d_iner', 'd_skel']:
            raise ValueError(f'Unsupported key {key} for matlab file')
        data = loadmat(file_path)[key]
        return data
    except Exception as e:
        logger.error(f"Error loading MAT {file_path}: {e}")
        return None


LOADER_MAP = {'csv': csvloader, 'mat': matloader}


def select_subwindow_pandas(unimodal_data):
    """Select subwindow with highest variance"""
    n = len(unimodal_data)
    magnitude = np.linalg.norm(unimodal_data, axis=1)
    df = pd.DataFrame({"values": magnitude})
    df["variance"] = df["values"].rolling(window=125).var()
    max_idx = df["variance"].idxmax()
    final_start = max(0, max_idx - 100)
    final_end = min(n, max_idx + 100)
    return unimodal_data[final_start:final_end, :]


def process_file_parallel_wrapper(args):
    """
    Wrapper for parallel processing that imports within the function
    """
    file_path, target_fs, viz_dir, trial_name, sampling_args = args
    
    # Import here to avoid circular import issues
    try:
        from utils.hybrid_interpolation import process_file_parallel
        return process_file_parallel(args)
    except ImportError as e:
        logger.error(f"Cannot import process_file_parallel: {e}")
        return None


class ParallelFileProcessor:
    """
    Optimized parallel file processing with robust 30Hz targeting
    """
    
    def __init__(self, max_workers=None, cores_per_file=4):
        """
        Args:
            max_workers: Maximum number of worker processes
            cores_per_file: Cores per file (for resource allocation)
        """
        total_cores = cpu_count()
        self.max_workers = max_workers or min(total_cores, 32)
        self.cores_per_file = cores_per_file
        self.effective_workers = max(1, self.max_workers // cores_per_file)
        
        logger.info(f"Parallel processor: {self.max_workers} cores, "
                   f"{self.cores_per_file} cores/file, {self.effective_workers} workers")
    
    def process_files_parallel(self, file_tasks, desc="Processing files"):
        """
        Process files in parallel with robust error handling
        """
        if len(file_tasks) == 0:
            return []
        
        # Single-threaded for small batches
        if len(file_tasks) <= 2 or self.effective_workers == 1:
            logger.info(f"Processing {len(file_tasks)} files single-threaded")
            results = []
            for file_path, kwargs in file_tasks:
                result = csvloader(file_path, **kwargs)
                results.append(result)
            return results
        
        # Multi-threaded processing
        logger.info(f"Processing {len(file_tasks)} files with {self.effective_workers} workers")
        
        # Prepare arguments for parallel processing
        parallel_args = []
        for file_path, kwargs in file_tasks:
            target_fs = kwargs.get('target_fs', 30)
            viz_dir = kwargs.get('viz_dir', None)
            trial_name = kwargs.get('trial_name', None)
            
            sampling_args = {
                'max_upsample_ratio': kwargs.get('max_upsample_ratio', 1.5),
                'min_samples': kwargs.get('min_samples', 10),
                'gap_threshold_seconds': kwargs.get('gap_threshold_seconds', 0.5)
            }
            
            parallel_args.append((file_path, target_fs, viz_dir, trial_name, sampling_args))
        
        # Process with timeout and error handling
        results = []
        try:
            with Pool(processes=self.effective_workers) as pool:
                results = pool.map(process_file_parallel_wrapper, parallel_args)
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing")
            # Fallback to sequential
            for args in parallel_args:
                try:
                    result = process_file_parallel_wrapper(args)
                    results.append(result)
                except Exception as sub_e:
                    logger.error(f"Sequential fallback failed for {args[0]}: {sub_e}")
                    results.append(None)
        
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Parallel processing complete: {successful}/{len(results)} successful")
        return results


class UTD_MM_TF(tf.keras.utils.Sequence):
    """TensorFlow data sequence with 30Hz optimized data loading"""
    
    def __init__(self, dataset, batch_size, use_smv=False, window_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_smv = use_smv
        self.window_size = window_size
        
        self.acc_data = dataset.get('accelerometer')
        self.skl_data = dataset.get('skeleton')
        self.labels = dataset.get('labels')
        
        self._validate_data()
        self.indices = np.arange(self.num_samples)
    
    def _validate_data(self):
        """Validate and prepare data tensors"""
        if self.acc_data is None or len(self.acc_data) == 0:
            self.acc_data = np.zeros((1, self.window_size, 3), dtype=np.float32)
            self.num_samples = 1
            logger.warning("No accelerometer data found, using dummy data")
        else:
            self.num_samples = len(self.acc_data)
        
        if self.skl_data is not None and len(self.skl_data) > 0:
            if len(self.skl_data.shape) == 3:
                self.skl_seq, self.skl_length, self.skl_features = self.skl_data.shape
                if self.skl_features == 96:
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, 32, 3)
                elif self.skl_features % 3 == 0:
                    joints = self.skl_features // 3
                    self.skl_data = self.skl_data.reshape(self.skl_seq, self.skl_length, joints, 3)
        else:
            self.skl_data = np.zeros((self.num_samples, self.window_size, 32, 3), dtype=np.float32)
        
        if self.labels is None:
            self.labels = np.zeros(self.num_samples, dtype=np.int32)
        
        # Convert to tensors
        self.acc_data = tf.convert_to_tensor(self.acc_data, dtype=tf.float32)
        self.skl_data = tf.convert_to_tensor(self.skl_data, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
    
    def cal_smv(self, sample):
        """Calculate Signal Magnitude Vector"""
        mean = tf.reduce_mean(sample, axis=-2, keepdims=True)
        zero_mean = sample - mean
        sum_squared = tf.reduce_sum(tf.square(zero_mean), axis=-1, keepdims=True)
        smv = tf.sqrt(sum_squared)
        return smv
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_data = {}
        batch_acc = tf.gather(self.acc_data, batch_indices)
        
        if self.use_smv:
            batch_smv = self.cal_smv(batch_acc)
            batch_data['accelerometer'] = tf.concat([batch_smv, batch_acc], axis=-1)
        else:
            batch_data['accelerometer'] = batch_acc
        
        batch_data['skeleton'] = tf.gather(self.skl_data, batch_indices)
        batch_labels = tf.gather(self.labels, batch_indices)
        
        return batch_data, batch_labels, batch_indices
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class DatasetBuilder:
    """Enhanced dataset builder with robust 30Hz processing"""
    
    def __init__(self, dataset, mode, max_length, task='fd', **kwargs):
        self.dataset = dataset
        self.data = defaultdict(list)
        self.kwargs = kwargs
        self.mode = mode
        self.max_length = max_length
        self.task = task
        self.fuse = False
        
        # Fixed 30Hz parameters
        self.sampling = kwargs.get('sampling', 'none')
        self.target_fs = 30  # Fixed at 30Hz
        self.viz_resampling = kwargs.get('viz_resampling', False)
        self.viz_resampling_dir = kwargs.get('viz_resampling_dir', 'viz/resampling_30hz')
        
        # Parallel processing
        self.use_parallel = kwargs.get('use_parallel', True)
        self.max_workers = kwargs.get('max_workers', None)
        self.cores_per_file = kwargs.get('cores_per_file', 4)
        
        # 30Hz interpolation parameters
        self.max_upsample_ratio = kwargs.get('max_upsample_ratio', 1.5)
        self.min_samples = kwargs.get('min_samples', 10)
        self.gap_threshold_seconds = kwargs.get('gap_threshold_seconds', 0.5)
        
        # Initialize parallel processor
        if self.use_parallel:
            self.parallel_processor = ParallelFileProcessor(
                max_workers=self.max_workers,
                cores_per_file=self.cores_per_file
            )
        else:
            self.parallel_processor = None
            
        logger.info(f"DatasetBuilder initialized for 30Hz processing")
    
    def load_file(self, file_path):
        """Load single file with robust 30Hz interpolation"""
        loader = self._import_loader(file_path)
        trial_name = os.path.splitext(os.path.basename(file_path))[0]
        
        viz_dir = self.viz_resampling_dir if self.viz_resampling else None
        
        load_kwargs = {
            **self.kwargs,
            'sampling': self.sampling,
            'target_fs': self.target_fs,
            'viz_dir': viz_dir,
            'trial_name': trial_name,
            'max_upsample_ratio': self.max_upsample_ratio,
            'min_samples': self.min_samples,
            'gap_threshold_seconds': self.gap_threshold_seconds
        }
        
        return loader(file_path, **load_kwargs)
    
    def load_files_parallel(self, file_paths):
        """Load multiple files in parallel"""
        if not self.use_parallel or not self.parallel_processor:
            return [self.load_file(fp) for fp in file_paths]
        
        # Prepare tasks
        file_tasks = []
        for file_path in file_paths:
            trial_name = os.path.splitext(os.path.basename(file_path))[0]
            viz_dir = self.viz_resampling_dir if self.viz_resampling else None
            
            kwargs = {
                'sampling': self.sampling,
                'target_fs': self.target_fs,
                'viz_dir': viz_dir,
                'trial_name': trial_name,
                'max_upsample_ratio': self.max_upsample_ratio,
                'min_samples': self.min_samples,
                'gap_threshold_seconds': self.gap_threshold_seconds
            }
            file_tasks.append((file_path, kwargs))
        
        return self.parallel_processor.process_files_parallel(file_tasks)
    
    def _import_loader(self, file_path):
        """Import appropriate loader"""
        file_type = file_path.split('.')[-1]
        if file_type not in LOADER_MAP:
            raise ValueError(f'Unsupported file type {file_type}')
        return LOADER_MAP[file_type]
    
    def process(self, data, label):
        """Process data with windowing"""
        try:
            if self.mode == 'avg_pool':
                processed = {}
                for key, value in data.items():
                    processed[key] = pad_sequence_tf(value, self.max_length)
                processed['labels'] = np.array([label])
                return processed
            elif self.mode == 'selective_window':
                return selective_sliding_window(data, self.max_length, label)
            else:
                return sliding_window(data, self.max_length, 32, label)
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def make_dataset(self, subjects, fuse):
        """Create dataset with 30Hz processing"""
        self.data = defaultdict(list)
        self.fuse = fuse
        
        # Group accelerometer files for parallel processing
        acc_files = []
        other_trials = []
        
        for trial in self.dataset.matched_trials:
            if trial.subject_id in subjects:
                if self.task == 'fd':
                    label = int(trial.action_id > 9)
                elif self.task == 'age':
                    label = int(trial.subject_id < 29 or trial.subject_id > 46)
                else:
                    label = trial.action_id - 1
                
                # Separate accelerometer files for parallel processing
                if 'accelerometer' in trial.files and self.sampling == 'hybrid':
                    acc_files.append((trial.files['accelerometer'], trial, label))
                else:
                    other_trials.append((trial, label))
        
        # Process accelerometer files in parallel
        successful_acc = 0
        if acc_files and self.use_parallel and self.parallel_processor:
            logger.info(f"Processing {len(acc_files)} accelerometer files in parallel for 30Hz")
            
            acc_file_paths = [item[0] for item in acc_files]
            acc_results = self.load_files_parallel(acc_file_paths)
            
            # Process results
            for (file_path, trial, label), result in zip(acc_files, acc_results):
                if result is not None:
                    successful_acc += 1
                    trial_data = {'accelerometer': result}
                    
                    # Load other modalities
                    for modality, other_path in trial.files.items():
                        if modality != 'accelerometer':
                            try:
                                other_data = self.load_file(other_path)
                                if other_data is not None:
                                    trial_data[modality] = other_data
                            except Exception as e:
                                logger.debug(f"Error loading {modality}: {e}")
                    
                    self._process_trial_data(trial_data, label)
        
        # Process remaining trials
        for trial, label in other_trials:
            trial_data = {}
            for modality, file_path in trial.files.items():
                try:
                    unimodal_data = self.load_file(file_path)
                    if unimodal_data is None:
                        continue
                    
                    # Apply filtering (30Hz for hybrid, 25Hz for others)
                    if modality == 'accelerometer':
                        filter_fs = self.target_fs if self.sampling == 'hybrid' else 25
                        unimodal_data = butterworth_filter(unimodal_data, cutoff=7.5, fs=filter_fs)
                        
                        if unimodal_data.shape[0] > 250:
                            unimodal_data = select_subwindow_pandas(unimodal_data)
                    
                    trial_data[modality] = unimodal_data
                    
                except Exception as e:
                    logger.debug(f"Error loading {modality} from {file_path}: {e}")
                    continue
            
            self._process_trial_data(trial_data, label)
        
        # Finalize dataset
        for key in self.data:
            if len(self.data[key]) > 0:
                self.data[key] = np.concatenate(self.data[key], axis=0)
            else:
                self.data[key] = np.array([])
        
        total_samples = len(self.data.get('labels', []))
        logger.info(f"Dataset created: {total_samples} samples "
                   f"({successful_acc} accelerometer files processed at 30Hz)")
    
    def _process_trial_data(self, trial_data, label):
        """Process individual trial data"""
        if not trial_data:
            return
        
        # Apply DTW alignment if multiple modalities
        if len(trial_data) > 1:
            trial_data = align_sequence_dtw(trial_data)
        
        # Process with windowing
        processed_data = self.process(trial_data, label)
        if processed_data and len(processed_data.get('labels', [])) > 0:
            for key, value in processed_data.items():
                self.data[key].append(value)
    
    def normalization(self, acc_mean=None, acc_std=None, skl_mean=None, skl_std=None, compute_stats_only=False):
        """Normalize data with statistics computation"""
        if compute_stats_only:
            stats = {}
            for key, value in self.data.items():
                if key == 'labels' or len(value) == 0:
                    continue
                
                num_samples, length = value.shape[:2]
                norm_data = value.reshape(num_samples * length, -1)
                scaler = StandardScaler()
                scaler.fit(norm_data)
                
                if key == 'accelerometer':
                    stats['acc_mean'] = scaler.mean_
                    stats['acc_std'] = scaler.scale_
                elif key == 'skeleton':
                    stats['skl_mean'] = scaler.mean_
                    stats['skl_std'] = scaler.scale_
            return stats
        
        normalized_data = {}
        for key, value in self.data.items():
            if key == 'labels':
                normalized_data[key] = value
                continue
            
            if len(value) == 0:
                normalized_data[key] = value
                continue
            
            num_samples, length = value.shape[:2]
            norm_data = value.reshape(num_samples * length, -1)
            
            if key == 'accelerometer' and acc_mean is not None and acc_std is not None:
                norm_data = (norm_data - acc_mean) / acc_std
            elif key == 'skeleton' and skl_mean is not None and skl_std is not None:
                norm_data = (norm_data - skl_mean) / skl_std
            else:
                scaler = StandardScaler()
                norm_data = scaler.fit_transform(norm_data)
            
            normalized_data[key] = norm_data.reshape(value.shape)
        
        return normalized_data


def prepare_smartfallmm_tf(arg):
    """Prepare SmartFallMM dataset with 30Hz processing"""
    from utils.dataset_sf import SmartFallMM
    
    root_paths = [
        os.path.join(os.getcwd(), 'data/smartfallmm'),
        os.path.join(os.path.dirname(os.getcwd()), 'data/smartfallmm'),
        '/mmfs1/home/sww35/data/smartfallmm',
        '/path/to/data/smartfallmm'
    ]
    
    data_dir = next((p for p in root_paths if os.path.exists(p)), None)
    if data_dir is None:
        raise ValueError(f"Data directory not found. Tried: {root_paths}")
    
    logger.info(f"Using data directory: {data_dir}")
    
    # Initialize dataset
    sm_dataset = SmartFallMM(root_dir=data_dir)
    sm_dataset.pipeline(
        age_group=arg.dataset_args['age_group'],
        modalities=arg.dataset_args['modalities'],
        sensors=arg.dataset_args['sensors']
    )
    
    # Prepare dataset arguments
    dataset_args = arg.dataset_args.copy()
    dataset_args.pop('mode', None)
    dataset_args.pop('max_length', None)
    dataset_args.pop('task', None)
    
    # Add 30Hz processing configuration
    if hasattr(arg, 'use_parallel'):
        dataset_args['use_parallel'] = arg.use_parallel
    if hasattr(arg, 'max_workers'):
        dataset_args['max_workers'] = arg.max_workers
    if hasattr(arg, 'cores_per_file'):
        dataset_args['cores_per_file'] = arg.cores_per_file
    
    # Add visualization
    if hasattr(arg, 'viz_resampling'):
        dataset_args['viz_resampling'] = arg.viz_resampling
    if hasattr(arg, 'viz_resampling_dir'):
        dataset_args['viz_resampling_dir'] = arg.viz_resampling_dir
    
    # Add 30Hz interpolation parameters
    dataset_args['max_upsample_ratio'] = getattr(arg, 'max_upsample_ratio', 1.5)
    dataset_args['min_samples'] = getattr(arg, 'min_samples', 10)
    dataset_args['gap_threshold_seconds'] = getattr(arg, 'gap_threshold_seconds', 0.5)
    
    # Create builder
    builder = DatasetBuilder(
        sm_dataset,
        arg.dataset_args['mode'],
        arg.dataset_args.get('max_length', 128),
        arg.dataset_args['task'],
        **dataset_args
    )
    
    return builder


def split_by_subjects_tf(builder, subjects, fuse=False, compute_stats_only=False, 
                        acc_mean=None, acc_std=None, skl_mean=None, skl_std=None):
    """Split dataset by subjects with 30Hz processing"""
    builder.make_dataset(subjects, fuse)
    return builder.normalization(
        acc_mean=acc_mean, acc_std=acc_std, 
        skl_mean=skl_mean, skl_std=skl_std, 
        compute_stats_only=compute_stats_only
    )
