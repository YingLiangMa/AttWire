import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance,ImageStat
import os
import cv2
from tqdm import tqdm
import pandas as pd
from helpers import score_iou, _make_box_pts

IMAGE_SIZE = 400
SCALE_VECTOR = [IMAGE_SIZE, IMAGE_SIZE, 2 * np.pi, IMAGE_SIZE, IMAGE_SIZE]

# Source codes from
# https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/utils/image.py

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  # TODO debug
  # print(masked_gaussian.shape)
  # print(masked_heatmap.shape)
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: 
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


blackImg = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype = "uint8")

# Read echo label csv file
df = pd.read_csv('./labelsEcho.csv', delimiter=',')
df.drop(["imageId"], axis=1, inplace=True)
labels = df.to_numpy()
labels = [label * SCALE_VECTOR for label in labels]
labels = np.stack(labels)

# Read Blob label csv file
bdf = pd.read_csv('./blob.csv', delimiter=',')

# Read echo label csv file
df = pd.read_csv('./vallabelsEcho1400.csv', delimiter=',')
df.drop(["imageId"], axis=1, inplace=True)
labels = df.to_numpy()
labels = [label * SCALE_VECTOR for label in labels]
labels = np.stack(labels)

# Read Blob label csv file
bdf = pd.read_csv('./blob_val1400.csv', delimiter=',')

for counter in range(0, 6):
    xy = labels[counter]
    xy_int = xy.astype(int)
    centerXY = xy_int[0:2]
    print(counter)
    blackImg = np.zeros((IMAGE_SIZE, IMAGE_SIZE,3), dtype = "uint8")
    blackImg = blackImg/255.0
    draw_umich_gaussian(blackImg[:,:,0],centerXY,30)

    
    # Create gaussian map for blobs
    imgname=(f"{(counter):04d}.jpg")
    blob_df =bdf.loc[bdf['imageId'] == imgname]
    for i in range(len(blob_df)):
        tx = int(blob_df["Center X"].iloc[i] * IMAGE_SIZE)
        ty = int(blob_df["Center Y"].iloc[i] * IMAGE_SIZE)
        strength = blob_df["Strength"].iloc[i]
        xy[0]=tx
        xy[1]=ty
        # print(xy, strength)
        if strength >0.7:
          draw_umich_gaussian(blackImg[:,:,2],xy,12)
        else:
          draw_umich_gaussian(blackImg[:,:,2],xy,8)
    outname = f"./valmasks/{counter:03d}.png"
    blackImg = blackImg*255
    cv2.imwrite(outname, blackImg)