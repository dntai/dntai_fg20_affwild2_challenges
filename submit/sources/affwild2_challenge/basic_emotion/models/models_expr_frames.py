from .utils import classification_blocks, classification_regression_blocks, lstm_blocks

from keras.models import Sequential, Input, Model
from keras.layers import Dropout, Flatten, Dense, Reshape, GlobalMaxPooling3D, GlobalAveragePooling3D, TimeDistributed, Activation, \
                         Convolution3D, MaxPooling3D, Flatten, GRU, BatchNormalization
from keras.optimizers import SGD
from keras.layers import LSTM, Bidirectional

def EXP_FRAMES_V0(model_base_fea, 
                  model_base_pre, 
                  nb_classes = 7, 
                  fc_finals = [512, 512], 
                  fc_dropout = [0.1, 0.1, 0.1]):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape, name = "input_1")
    image_input = Input(shape=model_base_fea.input_shape[1:], name = "input_2")
    
    x1 = TimeDistributed(model_base_pre, name = "time_distributed_1")(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    
    x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)), name = "lambda_1")(x1)
    x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)), name = "lambda_2")(x1)
    x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)), name = "lambda_3")(x1)

    print(x1_max)
    print(x2)
    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3], name = "concatenate_1")

    x_all  = classification_blocks(x_all, 
                          class_name      = "expr_frames",
                          nb_classes      = nb_classes, 
                          fc_finals       = fc_finals, 
                          fc_dropout      = fc_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_all])

    return model
# EXP_FRAMES_V0

def EXP_FRAMES_LSTM_V0(model_base_fea, 
                  model_base_pre, 
                  nb_classes = 7, 
                  fc_finals = [512, 512], 
                  fc_dropout = [0.1, 0.1, 0.1], 
                  lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape)
    image_input = Input(shape=model_base_fea.input_shape[1:])
    
    x1 = TimeDistributed(model_base_pre)(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    x4 = lstm_blocks(x1, 
                     lstm_cell=lstm_cell, 
                     lstm_layers=lstm_layers, 
                     lstm_dropout = lstm_dropout, 
                     lstm_recurrent_dropout = lstm_recurrent_dropout)

    x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)))(x1)

    print(x1_max)
    print(x2)

    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3, x4])

    x_all  = classification_blocks(x_all, 
                          class_name      = "expr_frames",
                          nb_classes      = nb_classes, 
                          fc_finals       = fc_finals, 
                          fc_dropout      = fc_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_all])

    return model
# EXP_FRAMES_LSTM_V0

def EXP_FRAMES_LSTM_NO_STAT_V0(model_base_fea, 
                  model_base_pre, 
                  nb_classes = 7, 
                  fc_finals = [512, 512], 
                  fc_dropout = [0.1, 0.1, 0.1], 
                  lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape)
    image_input = Input(shape=model_base_fea.input_shape[1:])
    
    x1 = TimeDistributed(model_base_pre)(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    x4 = lstm_blocks(x1, 
                     lstm_cell=lstm_cell, 
                     lstm_layers=lstm_layers, 
                     lstm_dropout = lstm_dropout, 
                     lstm_recurrent_dropout = lstm_recurrent_dropout)

    # x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)))(x1)
    # x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)))(x1)
    # x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)))(x1)

    # print(x1_max)
    print(x2)

    
    # x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3, x4])
    x_all  = concatenate([x2, x3, x4])

    x_all  = classification_blocks(x_all, 
                          class_name      = "expr_frames",
                          nb_classes      = nb_classes, 
                          fc_finals       = fc_finals, 
                          fc_dropout      = fc_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_all])

    return model
# EXP_FRAMES_LSTM_NO_STAT_V0

def EXP_VA_FRAMES_V0(model_base_fea, 
                     model_base_pre, 
                     nb_classes = 7,
                     fc_regre_finals = [512, 512], 
                     fc_regre_dropout = [0.1, 0.1, 0.1],
                     fc_class_finals = [512, 512], 
                     fc_class_dropout = [0.1, 0.1, 0.1]):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape)
    image_input = Input(shape=model_base_fea.input_shape[1:])
    
    x1 = TimeDistributed(model_base_pre)(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    
    x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)))(x1)

    print(x1_max)
    print(x2)
    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3])

    [x_emotion, x_arousal, x_valence, x_aroval]  = classification_regression_blocks(x_all, 
                                              class_name       = "expr_va_frames",
                                              nb_classes       = 7,
                                              fc_regre_finals  = [512, 512], 
                                              fc_regre_dropout = [0.1, 0.1, 0.1],
                                              fc_class_finals  = [512, 512], 
                                              fc_class_dropout = [0.1, 0.1, 0.1])
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_V0

def EXP_VA_FRAMES_LSTM_V0(model_base_fea, 
                     model_base_pre, 
                     nb_classes = 7,
                     fc_regre_finals = [512, 512], 
                     fc_regre_dropout = [0.1, 0.1, 0.1],
                     fc_class_finals = [512, 512], 
                     fc_class_dropout = [0.1, 0.1, 0.1], 
                     lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape)
    image_input = Input(shape=model_base_fea.input_shape[1:])
    
    x1 = TimeDistributed(model_base_pre)(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    x4 = lstm_blocks(x1, 
                     lstm_cell=lstm_cell, 
                     lstm_layers=lstm_layers, 
                     lstm_dropout = lstm_dropout, 
                     lstm_recurrent_dropout = lstm_recurrent_dropout)
    
    x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)))(x1)
    x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)))(x1)

    print(x1_max)
    print(x2)
    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3, x4])

    [x_emotion, x_arousal, x_valence, x_aroval]  = classification_regression_blocks(x_all, 
                                              class_name       = "expr_va_frames",
                                              nb_classes       = 7,
                                              fc_regre_finals  = [512, 512], 
                                              fc_regre_dropout = [0.1, 0.1, 0.1],
                                              fc_class_finals  = [512, 512], 
                                              fc_class_dropout = [0.1, 0.1, 0.1])
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_LSTM_V0

def EXP_VA_FRAMES_LSTM_NO_STAT_V0(model_base_fea, 
                     model_base_pre, 
                     nb_classes = 7,
                     fc_regre_finals = [512, 512], 
                     fc_regre_dropout = [0.1, 0.1, 0.1],
                     fc_class_finals = [512, 512], 
                     fc_class_dropout = [0.1, 0.1, 0.1], 
                     lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape)
    image_input = Input(shape=model_base_fea.input_shape[1:])
    
    x1 = TimeDistributed(model_base_pre)(block_input)
    x2 = model_base_fea(image_input)
    x3 = model_base_pre(image_input)
    x4 = lstm_blocks(x1, 
                     lstm_cell=lstm_cell, 
                     lstm_layers=lstm_layers, 
                     lstm_dropout = lstm_dropout, 
                     lstm_recurrent_dropout = lstm_recurrent_dropout)
    
    # x1_max = Lambda(lambda x1: K.reshape(K.max(x1, axis = 1), shape = (-1, 7)))(x1)
    # x1_min = Lambda(lambda x1: K.reshape(K.min(x1, axis = 1), shape = (-1, 7)))(x1)
    # x1_mean = Lambda(lambda x1: K.reshape(K.mean(x1, axis = 1), shape = (-1, 7)))(x1)

    # print(x1_max)
    print(x2)
    
    # x_all  = concatenate([x1_max, x1_min, x1_mean, x2, x3, x4])
    x_all  = concatenate([x2, x3, x4])

    [x_emotion, x_arousal, x_valence, x_aroval]  = classification_regression_blocks(x_all, 
                                              class_name       = "expr_va_frames",
                                              nb_classes       = 7,
                                              fc_regre_finals  = [512, 512], 
                                              fc_regre_dropout = [0.1, 0.1, 0.1],
                                              fc_class_finals  = [512, 512], 
                                              fc_class_dropout = [0.1, 0.1, 0.1])
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_LSTM_NO_STAT_V0