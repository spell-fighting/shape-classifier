import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

assert len(K.tensorflow_backend._get_available_gpus()) > 0

# dimensions of our images.
img_width, img_height = 32, 32

train_data_dir = 'dataset/train'
epochs = 12
batch_size = 128
classes = ["circle", "hourglass", "square", "star", "triangle"]
num_classes = 5

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes,
    color_mode='grayscale',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes,
    color_mode='grayscale',
    subset='validation'
)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples,
    callbacks=[tbCallBack]
)

model.save('./models/keras/model_{}.h5'.format(len(next(os.walk("./models/keras"))[2])))

loss, acc = model.evaluate_generator(validation_generator)

print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
