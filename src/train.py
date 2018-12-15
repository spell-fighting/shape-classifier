import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from generator import DataGenerator
from config import SIZE, CLASSES, IMAGES_PER_CLASS, RATIO, data_path

assert len(K.tensorflow_backend._get_available_gpus()) > 0

epochs = 20
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (1, SIZE, SIZE)
else:
    input_shape = (SIZE, SIZE, 1)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(len(CLASSES), activation='softmax'))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)

training_generator = DataGenerator(IMAGES_PER_CLASS, CLASSES, RATIO, data_path, batch_size=batch_size, dim=(SIZE, SIZE))
validation_generator = DataGenerator(IMAGES_PER_CLASS, CLASSES, RATIO, data_path, batch_size=batch_size,
                                     dim=(SIZE, SIZE), mode="validation")

model.fit_generator(
    training_generator,
    epochs=epochs,
    steps_per_epoch=training_generator.__len__(),
    validation_data=validation_generator,
    validation_steps=validation_generator.__len__(),
    callbacks=[tbCallBack]
)

model.save('../models/keras/model_{}.h5'.format(len(next(os.walk("../models/keras"))[2])))
