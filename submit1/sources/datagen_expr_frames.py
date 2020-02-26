from keras.preprocessing.image import Iterator as KerasIterator
import numpy as np

class DataGenerator(KerasIterator):
    """
    Keras data generator for emotion recognition data loader
    """
    def __init__(self, dataloader, 
                       
                       has_dynamic_blocks = False,
                       dynamic_blocks = [4, 8, 16, 32],

                       batch_size = 32, 
                       shuffle = True, 
                       preprocessing_image_fn = None, 
                       seed = None):
        """
        """
        self.dataloader = dataloader
        
        self.has_dynamic_blocks = has_dynamic_blocks
        self.dynamic_blocks = dynamic_blocks
        ([x_image,x_block], y)          = self.dataloader[0]
        self.x_image_shape    = x_image.shape
        self.x_block_shape    = x_block.shape
        self.y_shape    = y.shape if y is not None else None
        self.current_batch = np.zeros(batch_size) # save current batch for easy debug
        self.preprocessing_image_fn = preprocessing_image_fn
        super().__init__(len(self.dataloader), batch_size, shuffle, seed)
    # __init__

    def _get_batches_of_transformed_samples(self, index_array):
        self.current_batch = index_array
        # dynamic_blocks
        if self.has_dynamic_blocks == True:
            block_size = self.dynamic_blocks[np.random.randint(0, len(self.dynamic_blocks))]
            self.x_block_shape = (block_size,) + self.x_block_shape[1:]
            self.dataloader.set_blocks(block_size)
        # if

        # (height, width, channel)
        batch_x_image = np.zeros((len(index_array),) + self.x_image_shape, dtype=np.float32)
        batch_x_block = np.zeros((len(index_array),) + self.x_block_shape, dtype=np.float32)
        
        batch_y = None
        if self.y_shape is not None:
            batch_y = np.zeros((len(index_array),) + self.y_shape, dtype=np.float32) if self.y_shape is not None else None
            for idx, value in enumerate(index_array):
                [batch_x_image[idx, ::], batch_x_block[idx, ::]], batch_y[idx, ::] = self.dataloader[value]
            # for
        else:
            for idx, value in enumerate(index_array):
                [batch_x_image[idx, ::], batch_x_block[idx, ::]], _ = self.dataloader[value]
            # for
        # if
        if self.preprocessing_image_fn is not None: 
            batch_x_image = self.preprocessing_image_fn(batch_x_image)
            batch_x_block = self.preprocessing_image_fn(batch_x_block)
        # if

        output = ([batch_x_image, batch_x_block], batch_y)
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
# DataGenerator