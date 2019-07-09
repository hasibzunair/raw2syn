from __future__ import print_function, division
import scipy
from keras.models import load_model
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import keras
import pandas as pd

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import cv2


def load_img(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        img = img/127.5 - 1.
        return img



#Root directory of the project
ROOT_DIR = os.path.abspath(".")
MODEL_PATH = os.path.join(ROOT_DIR, "saved_model")
TEST_IMAGES = os.path.join(ROOT_DIR, "test_imgs")
g_AB = load_model(os.path.join(MODEL_PATH, 'g_AB.h5'), custom_objects={'InstanceNormalization':InstanceNormalization})
#print(g_AB.summary())

# Load an image from domain A
img = load_img("{}/0093.jpg".format(TEST_IMAGES))
print(img.shape)
# Make it 4D for inference
img_4d = np.expand_dims(img, axis=0)
print(img_4d.shape)

# Generate domain B image
syn_img = g_AB.predict(img_4d)
# Make it 3D
img_B = np.squeeze(syn_img, axis=0)

# Rescale images 0 - 1
img_B = 0.5 * img_B + 0.5

print(img_B.shape)
# Save it
plt.imsave("test.jpg", img_B)