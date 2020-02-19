from keras.models import Sequential, Input, Model
from keras.layers import Dropout, Flatten, Dense, Reshape, GlobalMaxPooling3D, GlobalAveragePooling3D, TimeDistributed, Activation, \
                         Convolution3D, MaxPooling3D, Flatten, GRU, BatchNormalization
from keras.optimizers import SGD
from keras.layers import LSTM, Bidirectional

def FRAMES_EXPR(model_base_fea, model_base_pre, classes = 7, 
              fc_finals = [512, 512], fc_dropout = [0.1, 0.0, 0.0]):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape)
    image_input = Input(shape=model_base_fea.input_shape[1:])

    
    x1 = TimeDistributed(model_base_pre)(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    
    print(x1)
    x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)))(x1)

    print(x1_max)
    print(x2)
    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3])
    print(x_all)
    
#     if fc_dropout[0] > 0: x_all = Dropout(fc_dropout[0])(x_all)
#     if fc_finals[0]>0: x_all = Dense(fc_finals[0], activation='relu', name='frames_fc1')(x_all)
#     if fc_dropout[1] > 0: x_all = Dropout(fc_dropout[1])(x_all)
#     if fc_finals[1]>0: x_all = Dense(fc_finals[1], activation='relu', name='frames_fc2')(x_all)
#     if fc_dropout[2] > 0: x_all = Dropout(fc_dropout[2])(x_all)
    
    x_all = Dense(classes, activation='softmax', name='frames_predictions')(x_all)
    
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_all])
    return model
# FRAMES_EXPR