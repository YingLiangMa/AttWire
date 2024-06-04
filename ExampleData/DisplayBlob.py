import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance,ImageStat
import os
import cv2
from tqdm import tqdm
import pandas as pd

IMAGE_SIZE = 400

# Read label csv file
df = pd.read_csv('./blob.csv', delimiter=',')

xy = np.array([0, 0])
for counter in range(0, 6):
    imgname=(f"{(counter):04d}.jpg")
    blob_df =df.loc[df['imageId'] == imgname]
    print(blob_df)
    img = cv2.imread(f"./images/{counter:03d}.jpg")
    img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    plt.imshow(img)
    for i in range(len(blob_df)):
        tx = int(blob_df["Center X"].iloc[i] * IMAGE_SIZE)
        ty = int(blob_df["Center Y"].iloc[i] * IMAGE_SIZE)
        strength = int(blob_df["Strength"].iloc[i] * 20)
        xy[0]=tx
        xy[1]=ty
        print(xy, strength)
        #if strength >0.7:
        #  draw_umich_gaussian(blackImg,xy,12)
        #else:
        #  draw_umich_gaussian(blackImg,xy,8)
        plt.plot(xy[0], xy[1], 'b+')
        circle1 = plt.Circle((xy[0], xy[1]), strength, color='r',fill=False)
        plt.gca().add_patch(circle1)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
