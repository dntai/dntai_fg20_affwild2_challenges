from .model_vggface2 import Vggface2_ResNet50_Base
from keras.layers import Flatten, Dropout, Dense
from keras.models import Model
from keras.regularizers import l2

global weight_decay
weight_decay = 1e-4

def VGGFace2_FineTuning_EmotionRecognition_V0(weights_path = None,
                                              nb_classes = 7,
                                              dropout = [0.2, 0.2],
                                              mode = "train"):
    """
    VGGFace2 FineTuning for Emotion Recognition
    """
    base_model = Vggface2_ResNet50_Base()

    x = base_model.output
    if dropout[0]>0: x = Dropout(dropout[0])(x)
    x = Dense(2048, activation='relu', name='features')(x)
    if dropout[1] > 0: x = Dropout(dropout[1])(x)
    x = Dense(nb_classes,
              activation='softmax',
              name='predictions',
              use_bias=False, trainable=True,
              kernel_initializer='orthogonal',
              kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=base_model.input, outputs=x)

    if weights_path is not None:
        model.load_weights(weights_path, by_name = True)

    return model
# VGGFace2_FineTuning_EmotionRecognition_V0

def VGGFace2_AffWild2_V0(nb_classes = 7,dropout = [0.1, 0.1], mode = "train"):
    """
    VGGFace2 FineTuning for Emotion Recognition
    """
    base_model = Vggface2_ResNet50_Base()

    x = base_model.output
    if dropout[0]>0: x = Dropout(dropout[0])(x)
    x = Dense(2048, activation='relu', name='features')(x)
    if dropout[1] > 0: x = Dropout(dropout[1])(x)
    x = Dense(nb_classes,
              activation='softmax',
              name='predictions',
              use_bias=False, trainable=True,
              kernel_initializer='orthogonal',
              kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model
# VGGFace2_FineTuning_EmotionRecognition_V0