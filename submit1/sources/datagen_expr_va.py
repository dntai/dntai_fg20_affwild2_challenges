from keras.preprocessing.image import Iterator as KerasIterator
import numpy as np

class VAExprDataGenerator(KerasIterator):
    """
    Keras data generator for emotion recognition data loader
    """
    def __init__(self, dataloader, batch_size = 32, shuffle = True, preprocessing_image_fn = None, seed = None):
        """
        """
        self.dataloader = dataloader
        (x, y)          = self.dataloader[0]
        self.x_shape    = x.shape
        self.y_emotion_shape = y[0].shape if y[0] is not None else None
        self.current_batch = np.zeros(batch_size) # save current batch for easy debug
        self.preprocessing_image_fn = preprocessing_image_fn
        super().__init__(len(self.dataloader), batch_size, shuffle, seed)
    # __init__

    def _get_batches_of_transformed_samples(self, index_array):
        self.current_batch = index_array

        # (height, width, channel)
        batch_x = np.zeros((len(index_array),) + self.x_shape, dtype=np.float32)

        batch_y = None
        if self.y_emotion_shape is not None:
            batch_y_emotion = np.zeros((len(index_array),) + self.y_emotion_shape, dtype=np.float32)
            batch_y_aro_ccc  = np.zeros((len(index_array),) + (1,), dtype=np.float32)
            batch_y_val_ccc  = np.zeros((len(index_array),) + (1,), dtype=np.float32)
            batch_y_valaro_mse  = np.zeros((len(index_array),) + (5,), dtype=np.float32)
            for idx, value in enumerate(index_array):
                batch_x[idx, ::], [batch_y_emotion[idx, ::], batch_y_aro_ccc[idx, 0], batch_y_val_ccc[idx, 0]] = self.dataloader[value]
            # for
            batch_y_valaro_mse[:, 0] = batch_y_aro_ccc[:, 0]
            batch_y_valaro_mse[:, 1] = batch_y_val_ccc[:, 0]
            batch_y_valaro_mse[:, 2] = (batch_y_valaro_mse[:, 0] + batch_y_valaro_mse[:, 1]) / 2.0
            batch_y_valaro_mse[:, 3] = abs((batch_y_valaro_mse[:, 0] - batch_y_valaro_mse[:, 2]))
            batch_y_valaro_mse[:, 4] = abs((batch_y_valaro_mse[:, 1] - batch_y_valaro_mse[:, 2]))
            batch_y = [batch_y_emotion, batch_y_aro_ccc, batch_y_val_ccc, batch_y_valaro_mse]
        else:
            for idx, value in enumerate(index_array):
                batch_x[idx, ::], _ = self.dataloader[value]
            # for
        # if
        if self.preprocessing_image_fn is not None: batch_x = self.preprocessing_image_fn(batch_x)

        output = (batch_x, batch_y)
        return output
        
    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    # next
# VAExprDataGenerator