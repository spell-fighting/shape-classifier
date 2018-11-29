import numpy as np
from keras import backend as K
import keras


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self,
                 number_of_samples,
                 labels,
                 ratio,
                 data_path,
                 batch_size=32,
                 dim=(28, 28),
                 n_channels=1,
                 shuffle=True,
                 mode="train"):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = len(labels)
        self.shuffle = shuffle
        self.data = []

        if mode == "train":
            self.offset = 0
            self.num_images_per_class = int(number_of_samples * ratio)
        else:
            self.offset = int(number_of_samples * ratio)
            self.num_images_per_class = int(number_of_samples * (1 - ratio))

        for label in labels:
            file = "{}/{}.npy".format(data_path, label)
            self.data.append(np.load(file).astype(np.float32))

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.num_images_per_class * self.n_classes / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        return self.__data_generation()

    def on_batch_end(self):
        pass

    def __data_generation(self):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization

        partition = int(self.batch_size / self.n_classes)

        indexes = np.random.randint(0, self.num_images_per_class, partition, np.int)

        x = np.empty((partition * self.n_classes, *self.dim, self.n_channels))
        y = np.empty(partition * self.n_classes, dtype=int)

        i = 0
        # Generate data
        for img_i in indexes:
            for class_x in range(self.n_classes):
                image = self.data[class_x][self.offset + (img_i % self.num_images_per_class)] / 255
                if K.image_data_format() == 'channels_first':
                    image = image.reshape((1, 28, 28))
                else:
                    image = image.reshape((28, 28, 1))

                x[i, ] = image
                y[i] = class_x
                i += 1


        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
