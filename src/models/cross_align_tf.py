# src/models/cross_align_tf.py
import tensorflow as tf

class CrossModalAligner(tf.keras.layers.Layer):
    """Cross-modal alignment using multi-head attention"""
    
    def __init__(self, feature_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=feature_dim // num_heads,
            dropout=0.0
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, student_features, teacher_features, training=False):
        """
        Args:
            student_features: Student model features [batch, seq_len, feature_dim]
            teacher_features: Teacher model features [batch, seq_len, feature_dim]
        Returns:
            aligned_features: Cross-aligned features [batch, seq_len, feature_dim]
        """
        # Multi-head attention with teacher as query, student as key/value
        aligned_output = self.cross_attention(
            query=teacher_features,
            key=student_features,
            value=student_features,
            training=training
        )
        
        # Add residual connection and normalize
        aligned_output = self.layer_norm(aligned_output + teacher_features)
        
        return aligned_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'num_heads': self.num_heads
        })
        return config
