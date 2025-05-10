import tensorflow as tf

class DistillationLoss:
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        
    def __call__(self, student_logits, teacher_logits, labels, teacher_features, student_features):
        # Ensure correct shapes
        student_logits = tf.reshape(student_logits, [-1])
        teacher_logits = tf.reshape(teacher_logits, [-1])
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)
        
        # Hard loss (BCE with logits)
        if self.pos_weight is not None:
            bce = tf.nn.weighted_cross_entropy_with_logits(
                labels=labels,
                logits=student_logits,
                pos_weight=self.pos_weight
            )
        else:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,
                logits=student_logits
            )
        hard_loss = tf.reduce_mean(bce)
        
        # Feature-based distillation
        if teacher_features is not None and student_features is not None:
            # Match PyTorch implementation
            teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
            correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
            weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
            
            # KL divergence between features
            soft_teacher = tf.nn.softmax(teacher_features / self.temperature, axis=-1)
            log_soft_student = tf.nn.log_softmax(student_features / self.temperature, axis=-1)
            
            # Expand weights for broadcasting
            weights = tf.expand_dims(tf.expand_dims(weights, -1), -1)
            
            kl_loss = tf.reduce_sum(soft_teacher * (tf.math.log(soft_teacher + 1e-10) - log_soft_student), axis=-1)
            kl_loss = tf.reduce_mean(weights * kl_loss) * (self.temperature ** 2)
            
            total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * hard_loss
        else:
            total_loss = hard_loss
            
        return total_loss


class BinaryFocalLoss:
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        
        # Calculate focal loss
        prob = tf.nn.sigmoid(y_pred)
        pt = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        
        focal_weight = alpha_t * tf.pow(1 - pt, self.gamma)
        ce_loss = -tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
        loss = focal_weight * ce_loss
        
        return tf.reduce_mean(loss)
