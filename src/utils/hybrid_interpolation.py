#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hybrid_interpolation.py - Robust 30Hz interpolation for variable-rate accelerometer data
Fixed to target exactly 30Hz with proper timestamp validation and no upsampling
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class RobustInterpolator:
    """
    Robust 30Hz interpolation that validates timestamps and prevents upsampling
    """

    def __init__(self, target_fs=30, viz_dir=None, max_upsample_ratio=1.5, 
                 min_samples=10, gap_threshold_seconds=0.5):
        """
        Args:
            target_fs: Target sampling frequency (Hz) - fixed at 30Hz
            viz_dir: Directory for visualization output
            max_upsample_ratio: Maximum allowed upsampling ratio (1.5 = 50% increase max)
            min_samples: Minimum number of samples required
            gap_threshold_seconds: Gap threshold in seconds (not multiplier)
        """
        self.target_fs = target_fs
        self.viz_dir = viz_dir
        self.max_upsample_ratio = max_upsample_ratio
        self.min_samples = min_samples
        self.gap_threshold_seconds = gap_threshold_seconds
        
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)

    def validate_and_parse_timestamps(self, timestamps):
        """
        Robust timestamp validation and parsing
        Returns None if timestamps are invalid
        """
        if len(timestamps) < self.min_samples:
            logger.warning(f"Too few timestamps: {len(timestamps)} < {self.min_samples}")
            return None
            
        # Convert to pandas Series for easier handling
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Check if all timestamps are NaN or empty
        if timestamps.isna().all() or timestamps.empty:
            logger.warning("All timestamps are NaN or empty")
            return None
        
        # Try multiple parsing strategies
        parsed_timestamps = None
        
        # Strategy 1: Already datetime objects
        if timestamps.dtype == 'datetime64[ns]' or hasattr(timestamps.iloc[0], 'timestamp'):
            try:
                if hasattr(timestamps.iloc[0], 'timestamp'):
                    parsed_timestamps = pd.to_datetime(timestamps, errors='coerce')
                else:
                    parsed_timestamps = timestamps
            except:
                pass
        
        # Strategy 2: Common formats with explicit format specification
        if parsed_timestamps is None:
            date_formats = [
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S.%f', 
                '%m/%d/%Y %H:%M:%S',
                '%Y%m%d %H:%M:%S.%f',
                '%Y%m%d %H:%M:%S'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_timestamps = pd.to_datetime(timestamps, format=fmt, errors='coerce')
                    if not parsed_timestamps.isna().all():
                        break
                except:
                    continue
        
        # Strategy 3: Automatic parsing as last resort
        if parsed_timestamps is None or parsed_timestamps.isna().all():
            try:
                # Suppress the warning by being more specific
                parsed_timestamps = pd.to_datetime(timestamps, infer_datetime_format=True, errors='coerce')
            except:
                logger.error("Failed to parse any timestamps")
                return None
        
        # Remove NaN timestamps
        valid_mask = ~parsed_timestamps.isna()
        if valid_mask.sum() < self.min_samples:
            logger.warning(f"Too few valid timestamps after parsing: {valid_mask.sum()} < {self.min_samples}")
            return None
        
        parsed_timestamps = parsed_timestamps[valid_mask]
        
        # Convert to seconds from first timestamp
        try:
            t_seconds = (parsed_timestamps - parsed_timestamps.iloc[0]).dt.total_seconds().values
        except Exception as e:
            logger.error(f"Error converting timestamps to seconds: {e}")
            return None
        
        # Ensure monotonic increasing
        if not np.all(np.diff(t_seconds) >= 0):
            logger.warning("Timestamps are not monotonic, sorting...")
            sort_idx = np.argsort(t_seconds)
            t_seconds = t_seconds[sort_idx]
            valid_mask = valid_mask[valid_mask].iloc[sort_idx].index
        
        # Check for reasonable duration
        duration = t_seconds[-1] - t_seconds[0]
        if duration <= 0:
            logger.warning("Zero or negative duration")
            return None
        
        if duration < 1.0:  # Less than 1 second
            logger.warning(f"Very short duration: {duration:.3f} seconds")
        
        return t_seconds, valid_mask

    def analyze_sampling_characteristics(self, t_seconds):
        """
        Analyze sampling characteristics with robust statistics
        """
        if len(t_seconds) < 2:
            return None
            
        intervals = np.diff(t_seconds)
        
        # Remove zero intervals (duplicates)
        intervals = intervals[intervals > 0]
        
        if len(intervals) == 0:
            logger.warning("No valid intervals found")
            return None
        
        # Robust statistics using percentiles
        mean_interval = np.median(intervals)
        std_interval = np.std(intervals)
        q25, q75 = np.percentile(intervals, [25, 75])
        iqr = q75 - q25
        
        # Coefficient of variation
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        
        # Estimated sampling rate
        avg_fs = 1.0 / mean_interval if mean_interval > 0 else 0
        
        # Detect gaps using absolute threshold in seconds
        gap_mask = intervals > self.gap_threshold_seconds
        gap_indices = np.where(gap_mask)[0]
        gap_durations = intervals[gap_mask]
        
        return {
            'avg_fs': avg_fs,
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'cv': cv,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'intervals': intervals,
            'gap_indices': gap_indices,
            'gap_durations': gap_durations if len(gap_durations) > 0 else np.array([]),
            'num_gaps': len(gap_indices),
            'gap_ratio': len(gap_indices) / len(intervals) if len(intervals) > 0 else 0
        }

    def determine_resampling_strategy(self, stats, duration):
        """
        Determine if we should resample and how
        """
        if stats is None:
            return 'skip', 'invalid_stats'
        
        original_fs = stats['avg_fs']
        target_samples = max(2, int(duration * self.target_fs))
        
        # Check if upsampling would exceed limit
        upsample_ratio = self.target_fs / original_fs
        
        if upsample_ratio > self.max_upsample_ratio:
            # Too much upsampling required - use original rate or downsample
            if original_fs > self.target_fs:
                return 'downsample', 'cubic'
            else:
                # Keep original rate
                effective_fs = original_fs
                target_samples = len(stats['intervals']) + 1
                logger.info(f"Keeping original rate {original_fs:.1f}Hz (would need {upsample_ratio:.1f}x upsampling)")
                return 'keep_original', 'none'
        
        # Reasonable resampling
        if stats['cv'] < 0.1 and stats['gap_ratio'] < 0.02:
            return 'resample', 'linear'
        elif stats['cv'] < 0.3 and stats['gap_ratio'] < 0.1:
            return 'resample', 'cubic'
        else:
            return 'resample', 'pchip'

    def interpolate_to_30hz(self, t_seconds, data, stats):
        """
        Interpolate data to exactly 30Hz with gap handling
        """
        duration = t_seconds[-1] - t_seconds[0]
        
        # Determine strategy
        strategy, method = self.determine_resampling_strategy(stats, duration)
        
        if strategy == 'skip':
            logger.warning(f"Skipping interpolation: {method}")
            return None, None, 'skipped'
        
        if strategy == 'keep_original':
            logger.info("Keeping original sampling rate")
            return t_seconds, data, 'original'
        
        # Calculate target samples for exactly 30Hz
        target_samples = max(2, int(duration * self.target_fs))
        t_uniform = np.linspace(0, duration, target_samples)
        
        # Handle gaps by segmenting data
        if stats['num_gaps'] > 0:
            return self._interpolate_with_gaps(t_seconds, data, t_uniform, stats, method)
        else:
            return self._interpolate_continuous(t_seconds, data, t_uniform, method)

    def _interpolate_continuous(self, t_seconds, data, t_uniform, method):
        """
        Interpolate continuous data (no significant gaps)
        """
        data_resampled = np.zeros((len(t_uniform), data.shape[1]))
        
        for axis in range(data.shape[1]):
            try:
                if method == 'linear':
                    f = interp1d(t_seconds, data[:, axis], kind='linear',
                               bounds_error=False, fill_value='extrapolate')
                elif method == 'cubic' and len(t_seconds) >= 4:
                    f = interp1d(t_seconds, data[:, axis], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
                elif method == 'pchip' and len(t_seconds) >= 3:
                    f = PchipInterpolator(t_seconds, data[:, axis])
                else:
                    # Fallback to linear
                    f = interp1d(t_seconds, data[:, axis], kind='linear',
                               bounds_error=False, fill_value='extrapolate')
                
                data_resampled[:, axis] = f(t_uniform)
                
            except Exception as e:
                logger.warning(f"Interpolation failed for axis {axis}, using linear fallback: {e}")
                f = interp1d(t_seconds, data[:, axis], kind='linear',
                           bounds_error=False, fill_value='extrapolate')
                data_resampled[:, axis] = f(t_uniform)
        
        return t_uniform, data_resampled, method

    def _interpolate_with_gaps(self, t_seconds, data, t_uniform, stats, method):
        """
        Interpolate data with gap handling
        """
        # Find gap positions in original data
        gap_indices = stats['gap_indices']
        
        # Create segments
        segments = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            # End segment before gap
            end_idx = gap_idx + 1
            if end_idx > start_idx and end_idx - start_idx >= 2:
                segments.append((start_idx, end_idx))
            start_idx = gap_idx + 1
        
        # Add final segment
        if start_idx < len(t_seconds):
            segments.append((start_idx, len(t_seconds)))
        
        # Initialize output
        data_resampled = np.zeros((len(t_uniform), data.shape[1]))
        
        # Interpolate each segment
        for start_idx, end_idx in segments:
            if end_idx - start_idx < 2:
                continue
                
            # Extract segment
            t_seg = t_seconds[start_idx:end_idx]
            data_seg = data[start_idx:end_idx]
            
            # Find corresponding uniform time indices
            start_t = t_seg[0]
            end_t = t_seg[-1]
            
            uniform_start = np.searchsorted(t_uniform, start_t)
            uniform_end = np.searchsorted(t_uniform, end_t, side='right')
            
            if uniform_end > uniform_start:
                t_uniform_seg = t_uniform[uniform_start:uniform_end]
                
                # Interpolate segment
                for axis in range(data.shape[1]):
                    try:
                        if method == 'pchip' and len(t_seg) >= 3:
                            f = PchipInterpolator(t_seg, data_seg[:, axis])
                        elif method == 'cubic' and len(t_seg) >= 4:
                            f = interp1d(t_seg, data_seg[:, axis], kind='cubic',
                                       bounds_error=False, fill_value='extrapolate')
                        else:
                            f = interp1d(t_seg, data_seg[:, axis], kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
                        
                        data_resampled[uniform_start:uniform_end, axis] = f(t_uniform_seg)
                        
                    except Exception as e:
                        logger.debug(f"Segment interpolation failed for axis {axis}: {e}")
                        # Linear fallback
                        f = interp1d(t_seg, data_seg[:, axis], kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
                        data_resampled[uniform_start:uniform_end, axis] = f(t_uniform_seg)
        
        return t_uniform, data_resampled, f'{method}_segmented'

    def visualize_comparison(self, timestamps_orig, data_original, t_uniform, data_resampled,
                           stats, method_used, trial_name):
        """
        Create visualization comparing original and resampled data
        """
        if not self.viz_dir:
            return

        try:
            fig, axes = plt.subplots(4, 2, figsize=(15, 12))
            fig.suptitle(f'30Hz Resampling - {trial_name}\nMethod: {method_used}', fontsize=14)

            # Plot each axis
            axis_labels = ['X', 'Y', 'Z']
            colors = ['blue', 'green', 'red']

            for i in range(3):
                # Full comparison
                ax = axes[i, 0]
                ax.scatter(timestamps_orig, data_original[:, i], alpha=0.6, s=8, 
                          color=colors[i], label='Original')
                ax.plot(t_uniform, data_resampled[:, i], color=colors[i], alpha=0.9, 
                       linewidth=1.5, label='30Hz Resampled')
                
                # Mark gaps if any
                if stats and len(stats['gap_indices']) > 0:
                    gap_times = timestamps_orig[stats['gap_indices']]
                    gap_values = data_original[stats['gap_indices'], i]
                    ax.scatter(gap_times, gap_values, color='red', s=50, 
                             marker='x', label='Gaps', zorder=10)
                
                ax.set_ylabel(f'{axis_labels[i]} Acceleration')
                ax.set_title(f'{axis_labels[i]}-axis (Full)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Zoomed view
                ax = axes[i, 1]
                zoom_duration = min(3, timestamps_orig[-1] * 0.5)
                mask_orig = timestamps_orig <= zoom_duration
                mask_resamp = t_uniform <= zoom_duration
                
                if np.any(mask_orig) and np.any(mask_resamp):
                    ax.scatter(timestamps_orig[mask_orig], data_original[mask_orig, i],
                              alpha=0.6, s=15, color=colors[i], label='Original')
                    ax.plot(t_uniform[mask_resamp], data_resampled[mask_resamp, i],
                           color=colors[i], alpha=0.9, linewidth=2, label='30Hz')
                
                ax.set_ylabel(f'{axis_labels[i]} Acceleration')
                ax.set_title(f'{axis_labels[i]}-axis (0-{zoom_duration:.1f}s)')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Sampling statistics
            ax = axes[3, 0]
            if stats and len(stats['intervals']) > 0:
                ax.hist(stats['intervals'] * 1000, bins=min(30, len(stats['intervals'])//2), 
                       alpha=0.7, color='purple', edgecolor='black')
                ax.axvline(stats['mean_interval'] * 1000, color='red', linestyle='--', 
                          label=f'Median: {stats["mean_interval"]*1000:.1f}ms')
                ax.axvline(self.gap_threshold_seconds * 1000, color='orange', linestyle='--',
                          label=f'Gap threshold: {self.gap_threshold_seconds*1000:.0f}ms')
            
            ax.set_xlabel('Sampling Interval (ms)')
            ax.set_ylabel('Count')
            if stats:
                ax.set_title(f'Intervals (CV: {stats["cv"]:.3f}, Gaps: {stats["num_gaps"]})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Signal magnitude comparison
            ax = axes[3, 1]
            mag_original = np.sqrt(np.sum(data_original**2, axis=1))
            mag_resampled = np.sqrt(np.sum(data_resampled**2, axis=1))

            ax.plot(timestamps_orig, mag_original, 'k-', alpha=0.6, linewidth=1, label='Original')
            ax.plot(t_uniform, mag_resampled, 'r-', alpha=0.8, linewidth=1.5, label='30Hz')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal Magnitude')
            ax.set_title('Signal Magnitude Vector')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            filename = os.path.join(self.viz_dir, f'{trial_name}_30hz_comparison.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            # Create statistics summary
            self._create_stats_summary(stats, method_used, trial_name, timestamps_orig, data_resampled)
            
        except Exception as e:
            logger.error(f"Error creating visualization for {trial_name}: {e}")
            plt.close('all')

    def _create_stats_summary(self, stats, method_used, trial_name, timestamps_orig, data_resampled):
        """Create a summary plot of sampling statistics"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Calculate actual achieved sampling rate
            if len(data_resampled) > 1:
                duration = timestamps_orig[-1] - timestamps_orig[0]
                achieved_fs = (len(data_resampled) - 1) / duration if duration > 0 else 0
            else:
                achieved_fs = 0

            max_gap = max(stats['gap_durations']) if len(stats['gap_durations']) > 0 else 0

            summary_text = f"""
30Hz Resampling Statistics for {trial_name}
{'='*50}

Method Selected: {method_used}
Target Frequency: {self.target_fs} Hz

Original Data:
  Samples: {len(timestamps_orig)}
  Duration: {timestamps_orig[-1] - timestamps_orig[0]:.2f} seconds
  Average Sampling Rate: {stats['avg_fs']:.2f} Hz
  Mean Interval: {stats['mean_interval']*1000:.1f} ms
  CV: {stats['cv']:.3f}

Gap Analysis:
  Number of Gaps: {stats['num_gaps']}
  Gap Ratio: {stats['gap_ratio']:.3f}
  Max Gap: {max_gap*1000:.0f} ms (threshold: {self.gap_threshold_seconds*1000:.0f} ms)

Resampled Data:
  Samples: {len(data_resampled)}
  Achieved Rate: {achieved_fs:.2f} Hz
  Resampling Ratio: {achieved_fs/stats['avg_fs']:.2f}x
  {"⚠️ UPSAMPLED" if achieved_fs > stats['avg_fs'] * self.max_upsample_ratio else "✓ APPROPRIATE"}
"""

            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8))
            ax.axis('off')

            plt.title('30Hz Resampling Statistics Summary', fontsize=14, fontweight='bold')

            filename = os.path.join(self.viz_dir, f'{trial_name}_30hz_stats.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating stats summary for {trial_name}: {e}")
            plt.close('all')


def process_file_parallel(args):
    """
    Parallel processing function for a single file with robust 30Hz targeting
    """
    file_path, target_fs, viz_dir, trial_name, sampling_args = args
    
    try:
        logger.info(f"Processing {trial_name} targeting {target_fs}Hz (PID: {os.getpid()})")
        
        # Robust CSV loading
        try:
            # Read file content to detect format
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            
            # Check if we have proper timestamp format
            has_timestamps = False
            for line in first_lines:
                if line and ('/' in line or '-' in line or ':' in line):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            # Try to parse first part as timestamp
                            pd.to_datetime(parts[0])
                            has_timestamps = True
                            break
                        except:
                            continue
            
            if not has_timestamps:
                logger.warning(f"No valid timestamps found in {file_path}, skipping")
                return None
            
            # Load CSV
            df = pd.read_csv(file_path, header=None, names=['timestamp', 'x', 'y', 'z'], 
                           on_bad_lines='skip')
            
            # Clean and validate
            df = df.dropna()
            
            # Ensure numeric data for x, y, z
            for col in ['x', 'y', 'z']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['x', 'y', 'z'])
            
            if len(df) < 10:
                logger.warning(f"Too few valid samples in {file_path}: {len(df)}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")
            return None

        # Create interpolator
        process_viz_dir = None
        if viz_dir:
            process_viz_dir = os.path.join(viz_dir, f"process_{os.getpid()}")
            
        interpolator = RobustInterpolator(
            target_fs=target_fs, 
            viz_dir=process_viz_dir,
            **sampling_args
        )
        
        # Validate and parse timestamps
        result = interpolator.validate_and_parse_timestamps(df['timestamp'])
        if result is None:
            logger.warning(f"Invalid timestamps in {file_path}, skipping")
            return None
        
        t_seconds, valid_mask = result
        
        # Extract corresponding data
        acc_data = df[['x', 'y', 'z']].values[valid_mask].astype(np.float32)
        
        # Analyze sampling characteristics
        stats = interpolator.analyze_sampling_characteristics(t_seconds)
        if stats is None:
            logger.warning(f"Cannot analyze sampling for {file_path}, skipping")
            return None
        
        # Interpolate to 30Hz
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

        # Calculate achieved sampling rate
        duration = t_seconds[-1] - t_seconds[0] if len(t_seconds) > 1 else 0
        achieved_fs = (len(data_resampled) - 1) / duration if duration > 0 else 0

        logger.info(f"Completed {trial_name}: {len(acc_data)} -> {len(data_resampled)} samples "
                   f"(Target: {target_fs}Hz, Achieved: {achieved_fs:.1f}Hz, Method: {method_used})")

        return data_resampled
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def resample_accelerometer_to_30hz(csv_file, target_fs=30, viz_dir=None, trial_name=None, **kwargs):
    """
    Main function to resample accelerometer data to exactly 30Hz
    """
    return process_file_parallel((csv_file, target_fs, viz_dir, trial_name, kwargs))
