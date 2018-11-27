import os
import errno
from config import CLASSES, data_path

try:
    os.mkdir(data_path)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

for CLASS in CLASSES:
    os.system('curl https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy --output {}/{}.npy'.format(CLASS, data_path, CLASS))
