from __future__ import print_function
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

model = load_model("./models/keras/model_{}.h5".format(len(next(os.walk("./models/keras/"))[2]) - 1))

img = load_img("dataset/test/circle_0.png", grayscale=True)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

img_width, img_height = 64, 64

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model.summary()

prediction = model.predict(x)

print(
    "Circle : {} \n Hourglass : {} \n Square : {} \n Star : {} \n Triangle : {} \n".format(int(prediction[0][0] * 100),
                                                                                           int(prediction[0][1] * 100),
                                                                                           int(prediction[0][2] * 100),
                                                                                           int(prediction[0][3] * 100),
                                                                                           int(prediction[0][4] * 100)))
