import tensorflow as tf
from tensorflow import keras


class Distiller(tf.keras.Model):
    '''
    Knowledge distillation
    '''
    def __init__(self, student  : keras.Model , teacher : keras.Model) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    def compile(self, optimizer : keras.optimizers, metrics : keras.metrics, 
                student_loss_fn : keras.losses, distillation_loss_fn: keras.losses, alpha : float, 
                temperature : float) -> None : 
        '''
        Sets values of key componets
        '''
        super().compile(optimizer=optimizer, metrics = metrics, loss = student_loss_fn)
        self.student.compile(optimizer=optimizer, metrics = metrics, loss = student_loss_fn)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    
    def computer_loss( 
            self, x : tf.Tensor = None , y : tf.Tensor = None , y_pred : tf.Tensor = None ,
            sample_weight : tf.Tensor = None, allow_empty: tf.Tensor = False 
    ):
        teacher_pred = self.teacher(x, training = False)
        student_loss = self.studen_loss_fn(y , y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax ( teacher_pred / self.temperature, axis = 1), 
            tf.nn.softmax( y_pred / self.temperature, axis = 1)
        ) * (self.temperature ** 2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        skl_data, acc_data = x
        return self.student(acc_data)

