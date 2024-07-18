import tensorflow as tf

class SaveStudentModelCallback(tf.keras.callbacks.Callback):
    '''
    Callback to save student model
    '''
    def __init__(self, model_to_save, filepath):
        super(SaveStudentModelCallback, self).__init__()
        self.model_to_save = model_to_save
        self.filepath = filepath
         
    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(self.filepath.format(epoch))