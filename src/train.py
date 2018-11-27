import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from generator import generator, Dataset
from config import SIZE, CLASSES

# assert len(K.tensorflow_backend._get_available_gpus()) > 0

epochs = 20
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (1, SIZE, SIZE)
else:
    input_shape = (SIZE, SIZE, 1)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('linear'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('linear'))
model.add(Dropout(0.5))

model.add(Dense(len(CLASSES)))

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

validation_data = Dataset("valid")
training_data = Dataset("train")

validation_generator = generator(batch_size, validation_data)
training_generator = generator(batch_size, training_data)

model.fit_generator(
    training_generator,
    steps_per_epoch=training_data.len(),
    epochs=epochs,
    validation_steps=validation_data.len(),
    validation_data=validation_generator,
    callbacks=[tbCallBack]
)

model.save('./models/keras/model_{}.h5'.format(len(next(os.walk("./models/keras"))[2])))

loss, acc = model.evaluate_generator(validation_generator)

print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
