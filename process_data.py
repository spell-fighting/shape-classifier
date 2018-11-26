from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

import os

from PIL import Image

IMG_SIZE = 32


def pilimshow(img):
    plt.figure(1)
    plt.imshow(img)
    plt.show()

def preprocess(x):
    x = x.convert("L")
    xm = x.size[0] + 1
    ym = x.size[1] + 1
    xM = yM = -1
    y = x
    for i in range(x.size[0]):
        for j in range(x.size[1]):
            r = x.getpixel((i, j))
            # r,g,b = x.getpixel((i,j))
            if r > 0:
                r = g = b = 255
                if i < xm:
                    xm = i
                if i > xM:
                    xM = i
                if j < ym:
                    ym = j
                if j > yM:
                    yM = j
            else:
                r = g = b = 0
            y.putpixel((i, j), (r))
    if xm < xM:
        xr = int(0.1 * float(xM - xm))
        yr = int(0.1 * float(yM - ym))
        # print(str(xr) + " "+str(yr))
        xm -= xr
        xM += xr
        ym -= yr
        yM += yr
        y = y.crop((xm, ym, xM, yM))
    y = y.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    # print(str(xm) + " "+str(ym)+" "+str(xM)+" " +str(yM))
    # print("im tansform="+str(y.size))
    # pilimshow(x)
    # pilimshow(y)
    return y


def process_folder(root, dst):
    for subdir, dirs, files in os.walk(root):
        for file in files:
            xname = os.path.join(subdir, file)
            x = Image.open(xname)
            y = preprocess(x)
            yname = xname.replace(root, dst)
            try:
                os.makedirs(os.path.dirname(yname))
            except:
                pass
            print(xname + " ==> " + yname)
            y.save(yname)


if __name__ == "__main__":
    process_folder(root="dataset/original", dst="dataset/train")
