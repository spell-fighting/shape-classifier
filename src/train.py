import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from generator import DataGenerator
from config import SIZE, CLASSES, IMAGES_PER_CLASS, RATIO, data_path

# assert len(K.tensorflow_backend._get_available_gpus()) > 0

epochs = 20
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (1, SIZE, SIZE)
else:
    input_shape = (SIZE, SIZE, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(CLASSES)))
model.add(Activation('softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
)

tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)

training_generator = DataGenerator(IMAGES_PER_CLASS, CLASSES, RATIO, data_path, batch_size=batch_size, dim=(SIZE, SIZE))
validation_generator = DataGenerator(IMAGES_PER_CLASS, CLASSES, RATIO, data_path, batch_size=batch_size, dim=(SIZE, SIZE), mode="validation")

training_samples = 50000
validation_samples = training_samples * RATIO

model.fit_generator(
    training_generator,
    epochs=epochs,
    steps_per_epoch=training_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    callbacks=[tbCallBack]
)

model.save('../models/keras/model_{}.h5'.format(len(next(os.walk("../models/keras"))[2])))
