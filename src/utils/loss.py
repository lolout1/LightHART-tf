import tensorflow as tf

class BinaryFocalLoss:
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    def __call__(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1])
        prob = tf.nn.sigmoid(y_pred)
        pt = tf.where(tf.equal(y_true, 1), prob, 1 - prob)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - pt, self.gamma)
        ce_loss = -tf.math.log(tf.clip_by_value(pt, 1e-8, 1.0))
        return tf.reduce_mean(focal_weight * ce_loss)

class DistillationLoss:
    def __init__(self, temperature=4.5, alpha=0.6, pos_weight=None):
        self.temperature = temperature
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.bce = BinaryFocalLoss(alpha=0.6)
        self.kl_div = tf.keras.losses.KLDivergence(reduction='none')
    def __call__(self, student_logits, teacher_logits, labels, teacher_features, student_features):
        student_logits = tf.reshape(student_logits, [-1])
        teacher_logits = tf.reshape(teacher_logits, [-1])
        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)
        hard_loss = self.bce(labels, student_logits)
        if teacher_features is not None and student_features is not None:
            teacher_pred = tf.cast(tf.sigmoid(teacher_logits) > 0.5, tf.float32)
            correct_mask = tf.cast(tf.equal(teacher_pred, labels), tf.float32)
            weights = (1.0/1.5) * correct_mask + (0.5/1.5) * (1.0 - correct_mask)
            soft_teacher = tf.nn.softmax(teacher_features / self.temperature, axis=-1)
            log_soft_student = tf.nn.log_softmax(student_features / self.temperature, axis=-1)
            weights = tf.expand_dims(tf.expand_dims(weights, -1), -1)
            kl_loss = tf.reduce_sum(soft_teacher * (tf.math.log(soft_teacher + 1e-10) - log_soft_student), axis=-1)
            kl_loss = tf.reduce_mean(weights * kl_loss) * (self.temperature ** 2)
            total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * hard_loss
        else:
            total_loss = hard_loss
        return total_loss
