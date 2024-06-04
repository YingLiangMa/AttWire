import os
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import imread, createCLAHE
import cv2
from tqdm import tqdm

from helpers import score_iou, _make_box_pts

IMAGE_SIZE = 400
SCALE_VECTOR = [IMAGE_SIZE, IMAGE_SIZE, 2 * np.pi, IMAGE_SIZE, IMAGE_SIZE]

# loading validation data
# Generate the name list of image files
imgnames = []
for counter in range(0, 6):
    imgnames.append(f"./Images/{counter:03d}.jpg")

im_array = []
for i in tqdm(imgnames):
    im = cv2.imread(i)
    im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # need to convert image into float32 type before trainning
    gray = gray/255.0
    im_array.append(gray)

images = np.stack(im_array)
print(images.shape)

# Read label csv file
df = pd.read_csv('./labelsEcho.csv', delimiter=',')
df.drop(["imageId"], axis=1, inplace=True)

labels = df.to_numpy()
labels = [label * SCALE_VECTOR for label in labels]
labels = np.stack(labels)

# Display some of images
for counter in range(0, 6):
    if counter%100 == 0:
        print(counter)
    image = Image.fromarray((images[counter]*255).astype(np.uint8))
    plt.imshow(image)
    data = labels[counter]
    data_int = data.astype(int)
    centerXY = data_int[0:2]
    if centerXY[0]>0:
        xy = _make_box_pts(*labels[counter])
        xxyy = np.vstack((xy,xy[0,:]))
        plt.plot(xxyy[:, 0], xxyy[:, 1], c="r")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

