######################################
### CHECKING GPU ON COLAB          ### 
######################################
def check_tensorflow_environment():
    from distutils.version import LooseVersion
    import warnings
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# check_gpu

######################################
### CHOOSE GPU OR CPU              ### 
######################################

def choose_keras_environment(gpus = ["0"], keras_backend = "tensorflow", verbose = 1): # gpus = ["-1"], ["0", "1"]
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(gpus) # run GPU 0
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"        # run GPU 0,1
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"          # don't run GPU
    os.environ['KERAS_BACKEND'] = keras_backend
    if verbose == 1:
        print("Environment GPUs:")
        print("+ Choose GPUs: ", ",".join(gpus))
        print("+ Keras backend: ", keras_backend)
    # if
# choose_keras_envs


# init_session
def init_session():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
# init_session