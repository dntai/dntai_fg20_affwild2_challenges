"""
IMPORT COMMON LIBRARIES
"""

# common
import os, sys, glob, tqdm, numpy as np, argparse
from math import ceil

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

# keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model, Model
from keras.utils import multi_gpu_model

# prlab
from prlab.keras.envs import check_tensorflow_environment, choose_keras_environment, init_session
from prlab.utils.model_reports import plot_confusion_matrix, print_summary, model_report, buffer_print_string