import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import os

class BaseDistillation:
    """Base class for knowledge distillation setup in TensorFlow."""
    
    def __init__(self, args):
        """Initialize with command-line arguments."""
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.teacher_model = None
        self.student_model = None
        self.dataset = None

    def load_dataset(self):
        """Load and preprocess dataset using feeder/make_dataset_tf.py."""
        try:
            from feeder.make_dataset_tf import UTD_MM_TF
            dataset = UTD_MM_TF(
                data_path=self.args.data_path,
                batch_size=self.args.batch_size,
                shuffle_buffer=self.args.shuffle_buffer
            )
            self.dataset = {
                'train': dataset.get_train_dataset(),
                'val': dataset.get_val_dataset(),
                'test': dataset.get_test_dataset()
            }
            self.logger.info("Dataset loaded and preprocessed successfully")
        except ImportError as e:
            self.logger.error(f"Failed to import UTD_MM_TF: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def build_teacher_model(self):
        """Build the teacher model architecture."""
        model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def load_teacher_model(self):
        """Load teacher model weights or train from scratch if not available."""
        weights_path = self.args.teacher_weights
        if os.path.exists(weights_path):
            try:
                self.teacher_model = models.load_model(weights_path)
                self.logger.info(f"Teacher model loaded from {weights_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load teacher model from {weights_path}: {e}")
                self.train_teacher_model()
        else:
            self.logger.info(f"Teacher weights not found at {weights_path}. Training from scratch.")
            self.train_teacher_model()

    def train_teacher_model(self):
        """Train the teacher model from scratch and save weights."""
        if self.dataset is None:
            self.load_dataset()
        
        self.teacher_model = self.build_teacher_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        self.teacher_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        try:
            self.teacher_model.fit(
                self.dataset['train'],
                epochs=self.args.num_epochs_teacher,
                validation_data=self.dataset['val'],
                callbacks=self._get_callbacks('teacher')
            )
            self.teacher_model.save(self.args.teacher_weights)
            self.logger.info(f"Teacher model trained and saved to {self.args.teacher_weights}")
        except Exception as e:
            self.logger.error(f"Error during teacher training: {e}")
            raise

    def _get_callbacks(self, model_type):
        """Return a list of callbacks for training."""
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"{self.args.checkpoint_dir}/{model_type}_best.h5",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
        return [checkpoint, early_stopping]
