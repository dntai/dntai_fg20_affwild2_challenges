from keras.layers import Flatten, Dropout, Dense
from keras.models import Model, Input
from keras.layers import LSTM, Bidirectional

def lstm_blocks(x, lstm_cell=1024, lstm_layers=2, lstm_dropout=0.1, lstm_recurrent_dropout = 0.1):
    # x: (batch_size, frames, features)
    if lstm_layers == 1:
       x = Bidirectional(LSTM(lstm_cell, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(x)
    else:
        for i in range(1, lstm_layers):
            x = Bidirectional(LSTM(lstm_cell, return_sequences=True, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(x)
        x = Bidirectional(LSTM(lstm_cell, return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(x)
    return x
    pass
# lstm_blocks

def classification_blocks(x, 
                          class_name = "expr_image",
                          nb_classes = 7, 
                          fc_finals = [512, 512], 
                          fc_dropout = [0.1, 0.1, 0.1]):
    if fc_dropout[0]>0: x = Dropout(fc_dropout[0], name = "dropout_1")(x)
    if fc_finals[0]>0:  x = Dense(fc_finals[0], activation='relu', name=f'{class_name}_feature1')(x)
    if fc_dropout[1]>0: x = Dropout(fc_dropout[1], name = "dropout_2")(x)
    if fc_finals[1]>0:  x = Dense(fc_finals[1], activation='relu', name=f'{class_name}_feature2')(x)
    if fc_dropout[2]>0: x = Dropout(fc_dropout[2], name = "dropout_3")(x)
    x = Dense(nb_classes, activation='softmax', name=f'{class_name}_predictions_class')(x)
    return x
# def classification_blocks

def classification_regression_blocks(x, 
                          class_name = "expr_image",
                          nb_classes = 7,
                          fc_regre_finals = [512, 512], 
                          fc_regre_dropout = [0.1, 0.1, 0.1],
                          fc_class_finals = [512, 512], 
                          fc_class_dropout = [0.1, 0.1, 0.1]):


    # Emotion Branch
    x_emotion = x
    if fc_class_dropout[0]>0: x_emotion = Dropout(fc_class_dropout[0])(x_emotion)
    if fc_class_finals[0]>0:  x_emotion = Dense(fc_class_finals[0], activation='relu', name=f'{class_name}_class_feature1')(x_emotion)
    if fc_class_dropout[1]>0: x_emotion = Dropout(fc_class_dropout[1])(x_emotion)
    if fc_class_finals[1]>0:  x_emotion = Dense(fc_class_finals[1], activation='relu', name=f'{class_name}_class_feature2')(x_emotion)
    if fc_class_dropout[2]>0: x_emotion = Dropout(fc_class_dropout[2])(x_emotion)
    x_emotion = Dense(nb_classes, activation='softmax', name=f'{class_name}_predictions_class')(x_emotion)

    # Arcousal Branch
    x_arousal = x
    if fc_regre_dropout[0]>0: x_arousal = Dropout(fc_regre_dropout[0])(x_arousal)
    if fc_regre_finals[0]>0:  x_arousal = Dense(fc_regre_finals[0], activation='relu', name=f'{class_name}_aro_feature1')(x_arousal)
    if fc_regre_dropout[1]>0: x_arousal = Dropout(fc_regre_dropout[1])(x_arousal)
    if fc_regre_finals[1]>0:  x_arousal = Dense(fc_regre_finals[1], activation='relu', name=f'{class_name}_aro_feature2')(x_arousal)
    if fc_regre_dropout[2]>0: x_arousal = Dropout(fc_regre_dropout[2])(x_arousal)
    x_arousal = Dense(1, activation='tanh', name=f'{class_name}_predictions_aro_ccc')(x_arousal)

    # Valence MSE Branch
    x_valence = x
    if fc_regre_dropout[0]>0: x_valence = Dropout(fc_regre_dropout[0])(x_valence)
    if fc_regre_finals[0]>0:  x_valence = Dense(fc_regre_finals[0], activation='relu', name=f'{class_name}_val_feature1')(x_valence)
    if fc_regre_dropout[1]>0: x_valence = Dropout(fc_regre_dropout[1])(x_valence)
    if fc_regre_finals[1]>0:  x_valence = Dense(fc_regre_finals[1], activation='relu', name=f'{class_name}_val_feature2')(x_valence)
    if fc_regre_dropout[2]>0: x_valence = Dropout(fc_regre_dropout[2])(x_valence)
    x_valence = Dense(1, activation='tanh', name=f'{class_name}_predictions_val_ccc')(x_valence)

    # ArcousalValenceMSE Branch
    x_aroval = x
    if fc_regre_dropout[0]>0: x_aroval = Dropout(fc_regre_dropout[0])(x_aroval)
    if fc_regre_finals[0]>0:  x_aroval = Dense(fc_regre_finals[0], activation='relu', name=f'{class_name}_aroval_feature1')(x_aroval)
    if fc_regre_dropout[1]>0: x_aroval = Dropout(fc_regre_dropout[1])(x_aroval)
    if fc_regre_finals[1]>0:  x_aroval = Dense(fc_regre_finals[1], activation='relu', name=f'{class_name}_aroval_feature2')(x_aroval)
    if fc_regre_dropout[2]>0: x_aroval = Dropout(fc_regre_dropout[2])(x_aroval)
    x_aroval = Dense(5, activation='tanh', name=f'{class_name}_predictions_aroval_mse')(x_aroval)

    return [x_emotion, x_arousal, x_valence, x_aroval]
# def classification_blocks