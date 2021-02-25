import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])