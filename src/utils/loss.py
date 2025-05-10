import tensorflow as tf

class DistillationLoss:
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
    def __call__(self, student_logits, teacher_logits, labels, teacher_features, student_features):
        student_logits = tf.squeeze(student_logits)
        teacher_logits = tf.squeeze(teacher_logits)
        labels = tf.cast(tf.squeeze(labels), tf.float32)
        
        hard_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=student_logits)
        if self.pos_weight is not None:
            hard_loss = hard_loss * (self.pos_weight * labels + (1 - labels))
        
        teacher_probs = tf.sigmoid(teacher_logits)
        teacher_pred = tf.cast(teacher_probs > 0.5, tf.float32)
        correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
        weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
        
        if teacher_features is not None and student_features is not None:
            teacher_feat_flat = tf.reshape(teacher_features, [tf.shape(teacher_features)[0], -1])
            student_feat_flat = tf.reshape(student_features, [tf.shape(student_features)[0], -1])
            
            teacher_probs_feat = tf.nn.softmax(teacher_feat_flat / self.temperature, axis=-1)
            student_log_probs_feat = tf.nn.log_softmax(student_feat_flat / self.temperature, axis=-1)
            
            feature_loss = tf.reduce_sum(
                teacher_probs_feat * (tf.math.log(teacher_probs_feat + 1e-10) - student_log_probs_feat), 
                axis=-1
            )
            feature_loss = feature_loss * (self.temperature ** 2)
            
            total_loss = self.alpha * tf.reduce_mean(weights * feature_loss) + (1.0 - self.alpha) * tf.reduce_mean(weights * hard_loss)
        else:
            total_loss = tf.reduce_mean(weights * hard_loss)
        
        return total_loss

class BinaryFocalLoss:
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    def __call__(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        prob = tf.nn.sigmoid(y_pred)
        pt = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - pt, self.gamma)
        loss = -focal_weight * tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
        return tf.reduce_mean(loss)
