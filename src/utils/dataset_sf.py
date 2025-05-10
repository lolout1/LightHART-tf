# src/utils/dataset_sf.py
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModalityFile:
    def __init__(self, subject_id, action_id, sequence_number, file_path):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.file_path = file_path


class Modality:
    def __init__(self, name):
        self.name = name
        self.files = []
    
    def add_file(self, subject_id, action_id, sequence_number, file_path):
        self.files.append(ModalityFile(subject_id, action_id, sequence_number, file_path))


class MatchedTrial:
    def __init__(self, subject_id, action_id, sequence_number):
        self.subject_id = subject_id
        self.action_id = action_id
        self.sequence_number = sequence_number
        self.files = {}
    
    def add_file(self, modality_name, file_path):
        self.files[modality_name] = file_path


class SmartFallMM:
    """SmartFallMM dataset class - matches PyTorch implementation"""
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.age_groups = {
            "old": {},
            "young": {}
        }
        self.matched_trials = []
        self.selected_sensors = {}
    
    def add_modality(self, age_group, modality_name):
        if age_group not in self.age_groups:
            raise ValueError(f"Invalid age group: {age_group}")
        self.age_groups[age_group][modality_name] = Modality(modality_name)
    
    def select_sensor(self, modality_name, sensor_name=None):
        self.selected_sensors[modality_name] = sensor_name
    
    def load_files(self):
        """Load files from dataset directory"""
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                if modality_name == "skeleton":
                    modality_dir = os.path.join(self.root_dir, age_group, modality_name)
                else:
                    sensor_name = self.selected_sensors.get(modality_name)
                    if sensor_name:
                        modality_dir = os.path.join(self.root_dir, age_group, modality_name, sensor_name)
                    else:
                        continue
                
                if not os.path.exists(modality_dir):
                    logger.warning(f"Directory not found: {modality_dir}")
                    continue
                
                # Load all CSV files
                for root, _, files in os.walk(modality_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            try:
                                # Parse filename format: S01_A01_T01.csv
                                subject_id = int(file[1:3])
                                action_id = int(file[4:6])
                                sequence_number = int(file[7:9])
                                file_path = os.path.join(root, file)
                                modality.add_file(subject_id, action_id, sequence_number, file_path)
                            except Exception as e:
                                logger.warning(f"Error parsing filename {file}: {e}")
    
    def match_trials(self):
        """Match files across modalities"""
        trial_dict = {}
        
        # Group files by (subject, action, sequence)
        for age_group, modalities in self.age_groups.items():
            for modality_name, modality in modalities.items():
                for file in modality.files:
                    key = (file.subject_id, file.action_id, file.sequence_number)
                    if key not in trial_dict:
                        trial_dict[key] = {}
                    trial_dict[key][modality_name] = file.file_path
        
        # Keep only trials with all required modalities
        required_modalities = list(self.age_groups['young'].keys())
        
        for key, files_dict in trial_dict.items():
            if all(mod in files_dict for mod in required_modalities):
                subject_id, action_id, sequence_number = key
                trial = MatchedTrial(subject_id, action_id, sequence_number)
                for modality_name, file_path in files_dict.items():
                    trial.add_file(modality_name, file_path)
                self.matched_trials.append(trial)
        
        logger.info(f"Matched {len(self.matched_trials)} trials")
    
    def pipeline(self, age_group, modalities, sensors):
        """Setup and process dataset pipeline"""
        for age in age_group:
            for modality in modalities:
                self.add_modality(age, modality)
                if modality == 'skeleton':
                    self.select_sensor('skeleton')
                else:
                    for sensor in sensors:
                        self.select_sensor(modality, sensor)
        
        self.load_files()
        self.match_trials()
