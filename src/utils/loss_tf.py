import tensorflow as tf

class BinaryFocalLossTF(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2.0, from_logits=True, name='binary_focal_loss'):
        super(BinaryFocalLossTF, self).__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        # Convert labels to float for calculations
        y_true = tf.cast(y_true, tf.float32)
        
        # Convert logits to probabilities if needed
        if self.from_logits:
            prob = tf.sigmoid(y_pred)
        else:
            prob = y_pred
            
        # Calculate pt (probability of true class)
        pt = tf.where(tf.equal(y_true, 1.0), prob, 1 - prob)
        
        # Calculate alpha_t
        alpha_t = tf.where(tf.equal(y_true, 1.0), self.alpha, 1 - self.alpha)
        
        # Calculate the focal loss
        focal_loss = -alpha_t * tf.pow(1 - pt, self.gamma) * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
        
        return tf.reduce_mean(focal_loss)

class DistillationLossTF(tf.keras.losses.Loss):
    def __init__(self, temperature=4.5, alpha=0.6, name='distillation_loss'):
        super(DistillationLossTF, self).__init__(name=name)
        self.temperature = temperature
        self.alpha = alpha
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        
    def call(self, inputs):
        student_logits, teacher_logits, labels, student_features, teacher_features, weights = inputs
        
        # Hard loss (binary cross entropy between student predictions and ground truth)
        label_loss = self.bce(labels, student_logits)
        
        # Feature-based distillation
        # Convert features to probabilities
        soft_teacher = tf.nn.softmax(teacher_features / self.temperature, axis=-1)
        soft_student = tf.nn.log_softmax(student_features / self.temperature, axis=-1)
        
        # Calculate correct predictions from teacher to weight the distillation
        teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
        weights = (1.0 / 1.5) * correct_mask + (0.5 / 1.5) * (1 - correct_mask)
        
        # Apply KL divergence with weights
        feature_loss = self.kl_loss(soft_teacher, soft_student)
        feature_loss = tf.multiply(tf.expand_dims(tf.expand_dims(weights, -1), -1), feature_loss)
        feature_loss = tf.reduce_mean(feature_loss)
        
        # Calculate combined loss
        loss = self.alpha * feature_loss + (1 - self.alpha) * label_loss
        
        return loss
