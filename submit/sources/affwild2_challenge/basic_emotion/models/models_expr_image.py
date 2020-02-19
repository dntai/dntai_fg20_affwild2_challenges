from .utils import classification_blocks, classification_regression_blocks

from keras.layers import Flatten, Dropout, Dense
from keras.models import Model, Input
from keras.regularizers import l2

global weight_decay
weight_decay = 1e-4

def EXPR_IMAGE_V0(model_base,
                  nb_classes = 7,
                  fc_finals = [512, 512], 
                  fc_dropout = [0.1, 0.1, 0.1]):
    
    # model_base --> get "features" layer from VGGFace2 Emotion
    image_input = Input(shape=model_base.input_shape[1:])

    x = model_base(image_input)
    x = classification_blocks(x, 
                          class_name    = "expr_image",
                          nb_classes      = nb_classes, 
                          fc_finals       = fc_finals, 
                          fc_dropout      = fc_dropout)
    model = Model(inputs=image_input, outputs=x)
    return model
    pass
# EXPR_IMAGE_V0

def EXPR_VA_IMAGE_V0(model_base,
                  nb_classes = 7,
                  fc_regre_finals = [512, 512], 
                  fc_regre_dropout = [0.1, 0.1, 0.1],
                  fc_class_finals = [512, 512], 
                  fc_class_dropout = [0.1, 0.1, 0.1]):
    # model_base --> get "features" layer from VGGFace2 Emotion
    image_input = Input(shape=model_base.input_shape[1:])

    x = model_base(image_input)

    [x_emotion, x_arousal, x_valence, x_aroval] = classification_regression_blocks(x,
                          class_name    = "expr_va_image",
                          nb_classes = 7,
                          fc_regre_finals = [512, 512], 
                          fc_regre_dropout = [0.1, 0.1, 0.1],
                          fc_class_finals = [512, 512], 
                          fc_class_dropout = [0.1, 0.1, 0.1])

    model = Model(inputs=image_input, outputs=[x_emotion, x_arousal, x_valence, x_aroval])
    return model
    pass
# EXPR_VA_IMAGE_V0