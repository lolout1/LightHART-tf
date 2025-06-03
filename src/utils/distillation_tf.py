import tensorflow as tf
import logging
logger = logging.getLogger('distillation-tf')

class EnhancedDistillationLoss:
    def __init__(self, temperature=3.5, alpha=0.3, beta=0.4, gamma=0.3, use_attention_transfer=True, use_hint_learning=True, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_attention_transfer = use_attention_transfer
        self.use_hint_learning = use_hint_learning
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        logger.info(f"Enhanced distillation loss: T={temperature}, α={alpha}, β={beta}, γ={gamma}")
    
    def attention_transfer_loss(self, student_features, teacher_features):
        student_attn = tf.nn.l2_normalize(tf.reduce_mean(tf.square(student_features), axis=-1), axis=-1)
        teacher_attn = tf.nn.l2_normalize(tf.reduce_mean(tf.square(teacher_features), axis=-1), axis=-1)
        return tf.reduce_mean(tf.square(student_attn - teacher_attn))
    
    def hint_learning_loss(self, student_features, teacher_features, hint_layer):
        if student_features.shape[-1] != teacher_features.shape[-1] and hint_layer:
            student_features = hint_layer(student_features)
        return tf.reduce_mean(tf.square(tf.nn.l2_normalize(student_features, axis=-1) - tf.nn.l2_normalize(teacher_features, axis=-1)))
    
    def __call__(self, student_logits, teacher_logits, labels, teacher_features, student_features, hint_layer=None):
        if len(tf.shape(student_logits)) > 1 and tf.shape(student_logits)[1] == 1:
            student_logits = tf.squeeze(student_logits, axis=1)
        if len(tf.shape(teacher_logits)) > 1 and tf.shape(teacher_logits)[1] == 1:
            teacher_logits = tf.squeeze(teacher_logits, axis=1)
        if len(tf.shape(labels)) > 1 and tf.shape(labels)[1] == 1:
            labels = tf.squeeze(labels, axis=1)
        labels = tf.cast(labels, tf.float32)
        teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
        weights = correct_mask + 0.5 * (1.0 - correct_mask)
        hard_loss = weights * self.bce(labels, student_logits)
        teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
        student_log_probs = tf.nn.log_softmax(student_logits / self.temperature)
        kd_loss = tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs), axis=-1) * (self.temperature ** 2)
        feature_loss = 0.0
        if len(tf.shape(teacher_features)) > 2:
            teacher_features = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
        if len(tf.shape(student_features)) > 2:
            student_features = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
        if self.use_attention_transfer:
            feature_loss += self.attention_transfer_loss(student_features, teacher_features)
        if self.use_hint_learning and hint_layer is not None:
            feature_loss += self.hint_learning_loss(student_features, teacher_features, hint_layer)
        return self.alpha * tf.reduce_mean(kd_loss) + self.beta * feature_loss + self.gamma * tf.reduce_mean(hard_loss)

class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.75, gamma=2.0, from_logits=True, reduction='mean', name='binary_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        prob = tf.nn.sigmoid(y_pred) if self.from_logits else y_pred
        p_t = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow((1 - p_t), self.gamma)
        loss = -focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
        if self.reduction == 'mean':
            return tf.reduce_mean(loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(loss)
        return loss
