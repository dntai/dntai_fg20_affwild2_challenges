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

def EXP_FRAMES_V1(model_base_fea, 
                  model_base_pre, 
                  nb_classes = 7, 
                  fc_finals = [512, 512], 
                  fc_dropout = [0.1, 0.1, 0.1]):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape, name = "input_1")
    image_input = Input(shape=model_base_fea.input_shape[1:], name = "input_2")
    
    print("block_input: ", block_input)
    print("image_input: ", image_input)
    
    
    x1 = TimeDistributed(model_base_pre, name = "time_distributed_1")(block_input)
    x2 = TimeDistributed(model_base_fea, name = "time_distributed_2")(block_input)
    x3 = model_base_fea(image_input)
    x4 = model_base_pre(image_input)
     
    print("x1: ", x1)
    print("x2: ", x2)
    print("x3: ", x3)
    print("x4: ", x4)
    
    x1_max = Lambda(lambda x: K.max(x, axis = 1), name = "lambda_2")(x1)
    x1_min = Lambda(lambda x: K.min(x, axis = 1), name = "lambda_3")(x1)
    x1_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_4")(x1)
    x2_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_1")(x2)

    print("x1_max: ", x1_max)
    print("x2_mean: ", x2_mean)
    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2_mean, x3, x4], name = "concatenate_1")

    print("x_all: ", x_all)

    x_output = classification_blocks(x_all, 
                                     class_name      = "expr_frames",
                                     nb_classes      = nb_classes, 
                                     fc_finals       = fc_finals, 
                                     fc_dropout      = fc_dropout)
        
    
    print("x_all: ", x_output)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_output])

    return model
# EXP_FRAMES_V1

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
                          class_name      = "expr_frames_lstm",
                          nb_classes      = nb_classes, 
                          fc_finals       = fc_finals, 
                          fc_dropout      = fc_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_all])

    return model
# EXP_FRAMES_LSTM_V0

def EXP_FRAMES_LSTM_V1(model_base_fea, 
                  model_base_pre, 
                  nb_classes = 7, 
                  fc_finals = [512, 512], 
                  fc_dropout = [0.1, 0.1, 0.1], 
                  lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape, name = "input_1")
    image_input = Input(shape=model_base_fea.input_shape[1:], name = "input_2")
    
    print("block_input: ", block_input)
    print("image_input: ", image_input)    
    
    x1 = TimeDistributed(model_base_pre, name = "time_distributed_1")(block_input)
    x2 = TimeDistributed(model_base_fea, name = "time_distributed_2")(block_input)
    x3 = model_base_fea(image_input)
    x4 = model_base_pre(image_input)
    x5 = lstm_blocks(x1, 
                     lstm_cell=lstm_cell, 
                     lstm_layers=lstm_layers, 
                     lstm_dropout = lstm_dropout, 
                     lstm_recurrent_dropout = lstm_recurrent_dropout)

    print("x1: ", x1)
    print("x2: ", x2)
    print("x3: ", x3)
    print("x4: ", x4)
    print("x5: ", x5)
    
    x1_max = Lambda(lambda x: K.max(x, axis = 1), name = "lambda_1")(x1)
    x1_min = Lambda(lambda x: K.min(x, axis = 1), name = "lambda_2")(x1)
    x1_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_3")(x1)
    x2_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_4")(x2)

    print("x1_max: ", x1_max)
    print("x2_mean: ", x2_mean)

    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2_mean, x3, x4, x5])

    print("x_all: ", x_all)
    
    x_output  = classification_blocks(x_all, 
                          class_name      = "expr_frames_lstm",
                          nb_classes      = nb_classes, 
                          fc_finals       = fc_finals, 
                          fc_dropout      = fc_dropout)
        
    print("x_output: ", x_output)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_output])

    return model
# EXP_FRAMES_LSTM_V1

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
                          class_name      = "expr_frames_lstm_nostat",
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
                                              fc_regre_finals  = fc_regre_finals, 
                                              fc_regre_dropout = fc_regre_dropout,
                                              fc_class_finals  = fc_class_finals, 
                                              fc_class_dropout = fc_class_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_V0

def EXP_VA_FRAMES_V1(model_base_fea, 
                     model_base_pre, 
                     nb_classes = 7,
                     fc_regre_finals = [512, 512], 
                     fc_regre_dropout = [0.1, 0.1, 0.1],
                     fc_class_finals = [512, 512], 
                     fc_class_dropout = [0.1, 0.1, 0.1]):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape, name = "input_1")
    image_input = Input(shape=model_base_fea.input_shape[1:], name = "input_2")
    
    print("block_input: ", block_input)
    print("image_input: ", image_input) 
    
    x1 = TimeDistributed(model_base_pre, name = "time_distributed_1")(block_input)
    x2 = TimeDistributed(model_base_fea, name = "time_distributed_2")(block_input)
    x3 = model_base_fea(image_input)
    x4 = model_base_pre(image_input)
     
    print("x1: ", x1)
    print("x2: ", x2)
    print("x3: ", x3)
    print("x4: ", x4)
    
    x1_max = Lambda(lambda x: K.max(x, axis = 1), name = "lambda_2")(x1)
    x1_min = Lambda(lambda x: K.min(x, axis = 1), name = "lambda_3")(x1)
    x1_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_4")(x1)
    x2_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_1")(x2)

    print("x1_max: ", x1_max)
    print("x2_mean: ", x2_mean)
    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2_mean, x3, x4], name = "concatenate_1")

    print("x_all: ", x_all)
    
    [x_emotion, x_arousal, x_valence, x_aroval]  = classification_regression_blocks(x_all, 
                                              class_name       = "expr_va_frames",
                                              nb_classes       = 7,
                                              fc_regre_finals  = fc_regre_finals, 
                                              fc_regre_dropout = fc_regre_dropout,
                                              fc_class_finals  = fc_class_finals, 
                                              fc_class_dropout = fc_class_dropout)
    
    print("x_emotion: ", x_emotion)
    print("x_arousal: ", x_arousal)
    print("x_valence: ", x_valence)
    print("x_aroval: ", x_aroval)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_V1

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
                                              class_name       = "expr_va_frames_lstm",
                                              nb_classes       = 7,
                                              fc_regre_finals  = fc_regre_finals, 
                                              fc_regre_dropout = fc_regre_dropout,
                                              fc_class_finals  = fc_class_finals, 
                                              fc_class_dropout = fc_class_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_LSTM_V0

def EXP_VA_FRAMES_LSTM_V1(model_base_fea, 
                     model_base_pre, 
                     nb_classes = 7,
                     fc_regre_finals = [512, 512], 
                     fc_regre_dropout = [0.1, 0.1, 0.1],
                     fc_class_finals = [512, 512], 
                     fc_class_dropout = [0.1, 0.1, 0.1], 
                     lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    from keras.layers import maximum, minimum, concatenate, Lambda
    import keras.backend as K

    block_input = Input(shape=model_base_pre.input_shape, name = "input_1")
    image_input = Input(shape=model_base_fea.input_shape[1:], name = "input_2")
    
    print("block_input: ", block_input)
    print("image_input: ", image_input)    
    
    x1 = TimeDistributed(model_base_pre, name = "time_distributed_1")(block_input)
    x2 = TimeDistributed(model_base_fea, name = "time_distributed_2")(block_input)
    x3 = model_base_fea(image_input)
    x4 = model_base_pre(image_input)
    x5 = lstm_blocks(x1, 
                     lstm_cell=lstm_cell, 
                     lstm_layers=lstm_layers, 
                     lstm_dropout = lstm_dropout, 
                     lstm_recurrent_dropout = lstm_recurrent_dropout)

    print("x1: ", x1)
    print("x2: ", x2)
    print("x3: ", x3)
    print("x4: ", x4)
    print("x5: ", x5)
    
    x1_max = Lambda(lambda x: K.max(x, axis = 1), name = "lambda_1")(x1)
    x1_min = Lambda(lambda x: K.min(x, axis = 1), name = "lambda_2")(x1)
    x1_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_3")(x1)
    x2_mean = Lambda(lambda x: K.mean(x, axis = 1), name = "lambda_4")(x2)

    print("x1_max: ", x1_max)
    print("x2_mean: ", x2_mean)

    
    x_all  = concatenate([x1_max, x1_min, x1_mean, x2_mean, x3, x4, x5])

    print("x_all: ", x_all)

    [x_emotion, x_arousal, x_valence, x_aroval]  = classification_regression_blocks(x_all, 
                                              class_name       = "expr_va_frames_lstm",
                                              nb_classes       = 7,
                                              fc_regre_finals  = fc_regre_finals, 
                                              fc_regre_dropout = fc_regre_dropout,
                                              fc_class_finals  = fc_class_finals, 
                                              fc_class_dropout = fc_class_dropout)
        
    print("x_emotion: ", x_emotion)
    print("x_arousal: ", x_arousal)
    print("x_valence: ", x_valence)
    print("x_aroval: ", x_aroval)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_LSTM_V1

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
                                              class_name       = "expr_va_frames_lstm_nostat",
                                              nb_classes       = 7,
                                              fc_regre_finals  = fc_regre_finals, 
                                              fc_regre_dropout = fc_regre_dropout,
                                              fc_class_finals  = fc_class_finals, 
                                              fc_class_dropout = fc_class_dropout)
        
    print(image_input)
    print(block_input)
    print(x_all)
    
    model = Model(inputs=[image_input, block_input], outputs = [x_emotion, x_arousal, x_valence, x_aroval])
    
    return model
# EXP_VA_FRAMES_LSTM_NO_STAT_V0