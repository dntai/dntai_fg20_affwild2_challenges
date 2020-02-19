from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Reshape, GlobalMaxPooling3D, GlobalAveragePooling3D, TimeDistributed, Activation, \
                         Convolution3D, MaxPooling3D, Flatten, GRU, BatchNormalization
from keras.optimizers import SGD
from keras.layers import LSTM, Bidirectional

def CNN_3D(model_base, classes = 7, fc_finals = [512, 512], fc_dropout = [0.1, 0.0, 0.0]):
    model = Sequential()
    model.add(TimeDistributed(model_base, input_shape=model_base.input_shape)) # (batch_size, frames, features)
    model.add(TimeDistributed(Dense(1024)))  # (batch_size, frames, features)
    model.add(TimeDistributed(Reshape((32, 32, 1)))) # (batch_size, frames, features)

    # 1st layer group
    model.add(Convolution3D(64, (3, 3, 3), padding='same', name='cnn3d_conv1_1', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(64, (3, 3, 3), padding='same', name='cnn3d_conv1_2', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='cnn3d_pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3), padding='same', name='cnn3d_conv2_1', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(128, (3, 3, 3), padding='same', name='cnn3d_conv2_2', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='cnn3d_pool2'))

    # 3nd layer group
    model.add(Convolution3D(256, (3, 3, 3), padding='same', name='cnn3d_conv3_1', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(256, (3, 3, 3), padding='same', name='cnn3d_conv3_2', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(256, (3, 3, 3), padding='same', name='cnn3d_conv3_3', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='cnn3d_pool3'))
    model.add(GlobalAveragePooling3D())
    # model.add(Flatten())
    # FC layers group
    if fc_dropout[0] > 0: model.add(Dropout(fc_dropout[0]))
    if fc_finals[0]>0: model.add(Dense(fc_finals[0], activation='relu', name='cnn3d_fc1'))
    if fc_dropout[1] > 0: model.add(Dropout(fc_dropout[1]))
    if fc_finals[1]>0: model.add(Dense(fc_finals[1], activation='relu', name='cnn3d_fc2'))
    if fc_dropout[2] > 0: model.add(Dropout(fc_dropout[2]))
    model.add(Dense(classes, activation='softmax', name='cnn3d_predictions'))
    return model
# CNN_3D

def SIMPLE_LSTM(model_base, classes = 7, fc_finals = [512, 512], fc_dropout = [0.1, 0.0, 0.0],
                lstm_cell=1024, lstm_layers=1, lstm_dropout=0.2, lstm_recurrent_dropout = 0.2):
    """
    input_shape = (sequence_length, features_length)
    """
    model = Sequential()
    model.add(TimeDistributed(model_base, input_shape=model_base.input_shape))  # (batch_size, frames, features)
    if lstm_layers == 1:
        model.add(Bidirectional(LSTM(lstm_cell, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
    else:
        for i in range(1, lstm_layers):
            model.add(Bidirectional(LSTM(lstm_cell, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
        model.add(Bidirectional(LSTM(lstm_cell, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
    if fc_dropout[0] > 0: model.add(Dropout(fc_dropout[0]))
    if fc_finals[0] > 0: model.add(Dense(fc_finals[0], activation='relu', name='lstm3d_fc1'))
    if fc_dropout[1] > 0: model.add(Dropout(fc_dropout[1]))
    if fc_finals[1] > 0: model.add(Dense(fc_finals[1], activation='relu', name='lstm3d_fc2'))
    if fc_dropout[2] > 0: model.add(Dropout(fc_dropout[2]))
    model.add(Dense(classes, activation='softmax', name='lstm3d_predictions'))
    return model
    pass
# SIMPLE_LSTM

def BILSTM_CNN_3D(model_base, classes=7, fc_finals=[512, 512], fc_dropout=[0.1, 0.0, 0.0],
                  lstm_cell=1024, lstm_layers=1, lstm_dropout=0.2, lstm_recurrent_dropout=0.2):

    model = Sequential()
    model.add(TimeDistributed(model_base, input_shape=model_base.input_shape))  # (batch_size, frames, features)
    for i in range(lstm_layers):
        model.add(Bidirectional(LSTM(lstm_cell, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)))
    model.add(TimeDistributed(Dense(1024)))  # (batch_size, frames, features)
    model.add(TimeDistributed(Reshape((32, 32, 1))))# (batch_size, frames, features)

    # 1st layer group
    model.add(Convolution3D(64, (3, 3, 3), padding='same', name='conv1_1', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(64, (3, 3, 3), padding='same', name='conv1_2', strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, (3, 3, 3), padding='same', name='conv2_1',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(128, (3, 3, 3), padding='same', name='conv2_2',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))
    # 3nd layer group
    model.add(Convolution3D(256, (3, 3, 3), padding='same', name='conv3_1',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(256, (3, 3, 3), padding='same', name='conv3_2',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(256, (3, 3, 3), padding='same', name='conv3_3',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))
    # 4nd layer group
    model.add(Convolution3D(512, (3, 3, 3), padding='same', name='conv4_1',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(512, (3, 3, 3), padding='same', name='conv4_2',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution3D(512, (3, 3, 3), padding='same', name='conv4_3',strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))
    model.add(GlobalAveragePooling3D())
    # FC layers group
    if fc_dropout[0]>0: model.add(Dropout(fc_dropout[0]))
    if fc_finals[0]>0: model.add(Dense(fc_finals[0], activation='relu', name='fc1'))
    if fc_dropout[1]>0: model.add(Dropout(fc_dropout[1]))
    if fc_finals[1]>0: model.add(Dense(fc_finals[1], activation='relu', name='fc2'))
    if fc_dropout[2]>0: model.add(Dropout(fc_dropout[2]))
    model.add(Dense(classes, activation='softmax', name='predictions'))
    return model
# CNN_3D_BATCHNORM_01