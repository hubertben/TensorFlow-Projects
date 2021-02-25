import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
#ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)
ys = np.array([-4, -1, 2, 5, 8, 11], dtype=float)



model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
print(model.predict([15.0]))
print(model.predict([20.0]))
print(model.predict([50.0]))
print(model.predict([99.0]))