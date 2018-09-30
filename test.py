from __future__ import print_function
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# img = load_img("dataset/test/triangle/qjtspokgkzydrzv123iu.png")
# x = img_to_array(img)
# x = x.reshape((1,) + x.shape)
#
# img_width, img_height = 150, 150
#
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

img_width, img_height = 150, 150


model = load_model('./models/model_0.h5')

test_datagen = ImageDataGenerator()
validation_data_dir = 'dataset/validation'
batch_size = 16
classes = ["circle", "hourglass", "square", "star", "triangle"]

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes
)

loss, acc = model.evaluate_generator(test_generator, verbose=1)

print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

# prediction = model.predict(x)
#
# print("Circle : {} \n Hourglass : {} \n Square : {} \n Star : {} \n Triangle : {} \n".format(prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3], prediction[0][4]))

