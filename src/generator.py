import numpy as np
import random
from keras import backend as K

from config import CLASSES, IMAGES_PER_CLASS, RATIO, SIZE, data_path


def generator(batch_size, dataset):
    batch_feature = np.zeros((batch_size, SIZE, SIZE, 1))
    batch_labels = np.zeros((batch_size, len(CLASSES)))

    while True:
        for i in range(batch_size):
            index = random.randint(0, dataset.len() - 1)
            feature, label_index = dataset.getitem(index)
            batch_feature[i] = feature
            batch_labels[i] = label_index
        yield batch_feature, batch_labels


class Dataset:
    def __init__(self, mode):
        self.num_classes = len(CLASSES)

        if mode == "train":
            self.offset = 0
            self.num_images_per_class = int(IMAGES_PER_CLASS * RATIO)

        else:
            self.offset = int(IMAGES_PER_CLASS * RATIO)
            self.num_images_per_class = int(IMAGES_PER_CLASS * (1 - RATIO))

        self.num_samples = self.num_images_per_class * self.num_classes

    def len(self):
        return self.num_samples

    def getitem(self, item):
        file = "{}/{}.npy".format(data_path, CLASSES[int(item / self.num_images_per_class)])
        image = np.load(file).astype(np.float32)[self.offset + (item % self.num_images_per_class)]
        image /= 255

        if K.image_data_format() == 'channels_first':
            reshaped_image = image.reshape((1, 28, 28))
        else:
            reshaped_image = image.reshape((28, 28, 1))

        return reshaped_image, int(item / self.num_images_per_class)
