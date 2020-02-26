import numpy as np
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, ToFloat, 
    Resize, Normalize, Rotate, RandomCrop, Crop, CenterCrop, DualTransform, PadIfNeeded, RandomCrop, 
    IAAFliplr, IAASuperpixels, VerticalFlip, RandomGamma, ElasticTransform, ImageOnlyTransform
)

def normal1_postprocessing(image):
    image = ((image + 1) / 2.0) * 255.0
    return image.astype(np.uint8)
# normal_preprocessing

def normal_postprocessing(image):
    image = image * 255.
    return image.astype(np.uint8)
# normal_postprocessing

def train_aug(image_size, p=1.0):
    return Compose([
        Resize(image_size + 36, image_size + 36),
        CenterCrop(image_size, image_size),
        # RandomCrop(image_size, image_size, p=1),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        # Resize(image_size+36, image_size+36),
	    # CenterCrop(image_size, image_size),
        # Rotate(limit=15),
        HorizontalFlip(p=0.5),
        PadIfNeeded(min_height=image_size, min_width=image_size, p=1)
    ], p=p)
# train_aug

def valid_aug(image_size, p=1.0):
    return Compose([
        Resize(image_size, image_size, p=1),
        PadIfNeeded(min_height=image_size, min_width=image_size, p=1)
    ], p=p)
# valid_aug

vggface2_mean = (91.4953, 103.8827, 131.0912) # BGR
def vggface2_preprocessing_input(x):
    x[..., 0] -= vggface2_mean[0]
    x[..., 1] -= vggface2_mean[1]
    x[..., 2] -= vggface2_mean[2]
    return x
# vggface2_preprocessing_input

def vggface2_postprocessing_input(x):
    x[..., 0] += vggface2_mean[0]
    x[..., 1] += vggface2_mean[1]
    x[..., 2] += vggface2_mean[2]
    return x
# vggface2_postprocessing_input

vggface1_mean = (91.4953, 103.8827, 131.0912)
def vggface1_preprocessing_input(x):
    x[..., 0] -= vggface1_mean[0]
    x[..., 1] -= vggface1_mean[1]
    x[..., 2] -= vggface1_mean[2]
    return x
# preprocessing_input_bgr

def nasnet_preprocess_input(x):
    """
    Image RGB
    tf: will scale pixels between -1 and 1, sample-wise.
    """
    x = x[...,::-1] # BGR --> RGB
    x = x / 127.5
    x -= 1.0
    return x
# nasnet_preprocess_input

def inceptionresnetv2_preprocess_input(x):
    """
    Image RGB
    tf: will scale pixels between -1 and 1, sample-wise.
    """
    x = x[...,::-1] # BGR --> RGB
    x = x / 127.5
    x -= 1.0
    return x
# inceptionresnetv2_preprocess_input

def xception_preprocess_input(x):
    """
    Image RGB
    tf: will scale pixels between -1 and 1, sample-wise.
    """
    x = x[..., ::-1]  # BGR --> RGB
    x = x / 127.5
    x -= 1.0
    return x
# xception_preprocess_input

densenet_mean = [0.485, 0.456, 0.406]
densenet_std = [0.229, 0.224, 0.225]
def densenet_preprocess_input(x):
    """
    torch: will scale pixels between 0 and 1
    and then will normalize each channel with respect to the
    ImageNet dataset.
    """
    x = x[...,::-1] # BGR --> RGB
    x /= 255.
    x[..., 0] -= densenet_mean[0]
    x[..., 1] -= densenet_mean[1]
    x[..., 2] -= densenet_mean[2]

    x[..., 0] /= densenet_std[0]
    x[..., 1] /= densenet_std[1]
    x[..., 2] /= densenet_std[2]
    return x
# densenet_preprocess_input

# def rafdb_train_aug(crop_size, image_size, p=1.0):
#     return Compose([
#         Resize(crop_size[0], crop_size[1], p=1),
#         RandomCrop(image_size[0], image_size[1], p = 1),
#         Rotate(limit = 45),
#         # ToFloat(max_value  = 255.0),
#         OneOf([
#             HorizontalFlip(p=0.5),
#             OneOf([
#                 RandomResizedCrop(limit=0.125),
#                 # HorizontalShear(origin_img_size, np.random.uniform(-0.07,0.07)),
#                 ShiftScaleRotateHeng(dx=0, dy=0, scale=1, angle=np.random.uniform(0, 10)),
#                 ElasticTransformHeng(grid=10, distort=np.random.uniform(0, 0.15)),
#             ]),
#             OneOf([
#                 BrightnessShift(np.random.uniform(-0.05, 0.05)),
#                 BrightnessMultiply(np.random.uniform(1-0.05, 1+0.05)),
#                 DoGama(np.random.uniform(1-0.05, 1+0.05)),
#             ])
#         ]),
#         PadIfNeeded(min_height=image_size[0], min_width=image_size[1], p=1),
# #        Normalize(mean=0.5, std=0.5, max_pixel_value = 1.0),
#     ], p=p)
# # tumor_preprocess