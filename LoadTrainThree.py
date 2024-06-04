import os
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras import backend as K
from tqdm import tqdm
import math
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Reshape,
    Flatten,
    Dense,
    concatenate,
    Attention,
    Conv2DTranspose
)

from helpers import score_iou, _make_box_pts

IMAGE_SIZE = 400
MASK_SIZE = 200
SCALE_VECTOR = [MASK_SIZE, MASK_SIZE, 2 * np.pi, MASK_SIZE, MASK_SIZE] # For angle, should be randian

Sigma=0.25
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy = DGaussxx.conj().T

# custom filter
def my_filter(shape, dtype=None):
    f = DGaussxx[...,np.newaxis,np.newaxis]
    f2 = DGaussxy[...,np.newaxis,np.newaxis]
    f = np.append(f,f2,axis=3)
    f3 = DGaussyy[...,np.newaxis,np.newaxis]
    f = np.append(f,f3,axis=3)
    # assert f.shape == shape
    return K.variable(f, dtype='float32')

Sigma=0.5
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx2 = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy2 = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy2 = DGaussxx2.conj().T

# custom filter
def my_filter2(shape, dtype=None):
    f = DGaussxx2[...,np.newaxis,np.newaxis]
    f2 = DGaussxy2[...,np.newaxis,np.newaxis]
    f = np.append(f,f2,axis=3)
    f3 = DGaussyy2[...,np.newaxis,np.newaxis]
    f = np.append(f,f3,axis=3)
    # assert f.shape == shape
    return K.variable(f, dtype='float32')

Sigma=0.75
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx3 = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy3 = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy3 = DGaussxx3.conj().T

# custom filter
def my_filter3(shape, dtype=None):
    f = DGaussxx3[...,np.newaxis,np.newaxis]
    f2 = DGaussxy3[...,np.newaxis,np.newaxis]
    f = np.append(f,f2,axis=3)
    f3 = DGaussyy3[...,np.newaxis,np.newaxis]
    f = np.append(f,f3,axis=3)
    # assert f.shape == shape
    return K.variable(f, dtype='float32')

Sigma=1
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx4 = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy4 = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy4 = DGaussxx4.conj().T

# custom filter
def my_filter4(shape, dtype=None):
    f = DGaussxx4[...,np.newaxis,np.newaxis]
    f2 = DGaussxy4[...,np.newaxis,np.newaxis]
    f = np.append(f,f2,axis=3)
    f3 = DGaussyy4[...,np.newaxis,np.newaxis]
    f = np.append(f,f3,axis=3)
    # assert f.shape == shape
    return K.variable(f, dtype='float32')

Sigma=1.25
# Make kernel coordinates
X, Y = np.meshgrid(np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1), indexing='ij')

# Build the gaussian 2nd derivatives filters
DGaussxx5 = 1/(2*np.pi*Sigma**4)*(X**2/Sigma**2 - 1)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussxy5 = (1/(2*np.pi*Sigma**6))*(X*Y)*np.exp(-(X**2 + Y**2)/(2*Sigma**2))
DGaussyy5 = DGaussxx5.conj().T

# custom filter
def my_filter5(shape, dtype=None):
    f = DGaussxx5[...,np.newaxis,np.newaxis]
    f2 = DGaussxy5[...,np.newaxis,np.newaxis]
    f = np.append(f,f2,axis=3)
    f3 = DGaussyy5[...,np.newaxis,np.newaxis]
    f = np.append(f,f3,axis=3)
    # assert f.shape == shape
    return K.variable(f, dtype='float32')

def generate_model():
    input_tensor = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,1))
    filterout = Conv2D(filters=3, 
                      kernel_size = 3,
                      trainable=False,
                      kernel_initializer=my_filter,
                      padding='same') (input_tensor)
    filterout2 = Conv2D(filters=3, 
                      kernel_size = 5,
                      trainable=False,
                      kernel_initializer=my_filter2,
                      padding='same') (input_tensor)
    filterout3 = Conv2D(filters=3, 
                      kernel_size = 5,
                      trainable=False,
                      kernel_initializer=my_filter3,
                      padding='same') (input_tensor)
    filterout4 = Conv2D(filters=3, 
                      kernel_size = 7,
                      trainable=False,
                      kernel_initializer=my_filter4,
                      padding='same') (input_tensor)
    filterout5 = Conv2D(filters=3, 
                      kernel_size = 9,
                      trainable=False,
                      kernel_initializer=my_filter5,
                      padding='same') (input_tensor)
    
    merged = concatenate([filterout, filterout2,filterout3,filterout4,filterout5], axis=3)
    batchout = BatchNormalization()(merged)
    maxout1 = MaxPool2D()(batchout)

    # Second input with random filters
    randout = Conv2D(15,kernel_size=3, use_bias=True, padding="same", activation="relu")(input_tensor)
    batchout = BatchNormalization()(randout)
    maxout2 = MaxPool2D()(batchout)

    attention_Layer = Attention(use_scale=False,dropout=0.1)([maxout1, maxout2])
    merged = concatenate([maxout1, maxout2,attention_Layer], axis=3)

    filterout = Conv2D(64,kernel_size=3, use_bias=True, padding="same", activation="relu")(merged)
    batchout = BatchNormalization()(filterout)
    maxout = MaxPool2D()(batchout)

    filterout = Conv2D(128,kernel_size=3, use_bias=True, padding="same", activation="relu")(maxout)
    batchout = BatchNormalization()(filterout)
    maxout = MaxPool2D()(batchout)

    filterout = Conv2D(256,kernel_size=3, use_bias=True, padding="same", activation="relu")(maxout)
    batchout = BatchNormalization()(filterout)
    maxout = MaxPool2D()(batchout)

    # upsampling layers
    upconv = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(maxout)
    out = Conv2D(256, (3, 3), activation='relu', padding='same')(upconv)

    upconv = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(out)
    out = Conv2D(128, (3, 3), activation='relu', padding='same')(upconv)
    
    upconv = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(out)
    allout = Conv2D(256,kernel_size=3, use_bias=True, padding="same", activation="relu")(upconv)
    
    # output heatmap for center, if two class change to Conv2D(2,kernel_size=1,...)
    filterout = Conv2D(256,kernel_size=3, use_bias=True, padding="same", activation="relu")(allout)
    batchout = BatchNormalization()(filterout)
    out = Conv2D(256,kernel_size=1, use_bias=True, padding="same", activation="relu")(batchout)
    CenOut = Conv2D(3,kernel_size=1, use_bias=True, padding="same", activation='sigmoid', name = "cen_out")(out)
  
    # output heatmap for width,height
    filterout = Conv2D(256,kernel_size=3, use_bias=True, padding="same", activation="relu")(allout)
    batchout = BatchNormalization()(filterout)
    out = Conv2D(256,kernel_size=1, use_bias=True, padding="same", activation="relu")(batchout)
    SizeOut = Conv2D(2,kernel_size=1, use_bias=True, padding="same", activation='sigmoid', name = "size_out")(out)

    # output heatmap for angles
    filterout = Conv2D(256,kernel_size=3, use_bias=True, padding="same", activation="relu")(allout)
    batchout = BatchNormalization()(filterout)
    out = Conv2D(256,kernel_size=1, use_bias=True, padding="same", activation="relu")(batchout)
    AngleOut = Conv2D(1,kernel_size=1, use_bias=True, padding="same", activation='sigmoid', name = "angle_out")(out)

    model = Model(inputs=input_tensor, outputs=[CenOut, SizeOut, AngleOut])
    return model

class DetMetrics(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.totalloss = []

    def on_epoch_end(self, epoch, logs=None):
        floss = logs['cen_out_loss']
        mloss = logs['size_out_loss']
        aloss = logs['angle_out_loss']

        tloss = floss + mloss + aloss

        print(f" total loss: {tloss:.5f}")
        if self.totalloss and tloss < min(self.totalloss):
            self.model.save(self.model.best_model_path)
            print(f"\n\tnew best model save to {self.model.best_model_path}\n")

        self.totalloss.append(tloss)


def FocalLoss(y_true, y_pred):    
    beta = 0.8
    gamma = 2

    # inputs = K.flatten(y_true)
    # targets = K.flatten(y_pred)
    # Add small value to fix the issue (loss function return Nan)
    # https://stackoverflow.com/questions/72759580/loss-returns-nan-in-tensorflow
    f_loss = beta * (1 - y_pred+1e-10) ** gamma * y_true * K.log(y_pred+1e-10)  # β*(1-p̂)ᵞ*p*log(p̂)
    f_loss += (1 - beta) * (y_pred+1e-10) ** gamma * (1 - y_true) * K.log(1 - y_pred+1e-10)  # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
    f_loss = -f_loss  # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

    # Average over each data point/image in batch
    axis_to_reduce = range(1, K.ndim(f_loss))
    f_loss = K.mean(f_loss, axis=axis_to_reduce)
    
    return f_loss


model = generate_model()
model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), 
              loss={
                  'cen_out': FocalLoss,
                  'size_out': "mae",
                  'angle_out': "mae"})
model.summary()

# loading validation data
# Generate the name list of image files
imgnames = []
masknames = []
for counter in range(0, 1400):
    imgnames.append(f"./valimages/{counter:03d}.jpg")
    masknames.append(f"./valmasks/{counter:03d}.png")

im_array = []
for i in tqdm(imgnames):
    im = cv2.imread(i)
    im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = gray/255.0
    im_array.append(gray)

mask_array = []
for i in tqdm(masknames):
    im = cv2.imread(i)
    im = cv2.resize(im, dsize=(MASK_SIZE, MASK_SIZE))
    mask = im
    mask = mask/255.0
    mask_array.append(mask)

val_images = np.stack(im_array)
print(val_images.shape)

val_masks = np.stack(mask_array)
print(val_masks.shape)

# Create Size and Angle data
# Read label csv file
df = pd.read_csv('./vallabelsEcho1400.csv', delimiter=',')
df.drop(["imageId"], axis=1, inplace=True)
labels = df.to_numpy()
labels = [label * SCALE_VECTOR for label in labels]
labels = np.stack(labels)

# Read valve label csv file
valvedf = pd.read_csv('./vallabelsValve1400.csv', delimiter=',')
valvedf.drop(["imageId"], axis=1, inplace=True)
valvelabels = valvedf.to_numpy()
valvelabels = [label * SCALE_VECTOR for label in valvelabels]
valvelabels = np.stack(valvelabels)

# Read Blob label csv file
bdf = pd.read_csv('./blob_val1400.csv', delimiter=',')

smask_array = []
for counter in range(0, 1400):
    xy = labels[counter]
    xy_int = xy.astype(int)
    centerXY = xy_int[0:2]
    mask = np.zeros((MASK_SIZE, MASK_SIZE, 2), dtype = "float")
    # add echo data
    if centerXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                mask[centerXY[1]+i,centerXY[0]+j, 0] = xy[3]/MASK_SIZE
                mask[centerXY[1]+i,centerXY[0]+j, 1] = xy[4]/MASK_SIZE 
    # add valve size data
    vxy = valvelabels[counter]
    vxy_int = vxy.astype(int)
    vcenterXY = vxy_int[0:2]
    if vcenterXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                mask[vcenterXY[1]+i,vcenterXY[0]+j, 0] = vxy[3]/MASK_SIZE
                mask[vcenterXY[1]+i,vcenterXY[0]+j, 1] = vxy[4]/MASK_SIZE
    # Add blob data
    imgname=(f"{(counter):04d}.jpg")
    blob_df =bdf.loc[bdf['imageId'] == imgname]
    for i in range(len(blob_df)):
        tx = int(blob_df["Center X"].iloc[i] * MASK_SIZE)
        ty = int(blob_df["Center Y"].iloc[i] * MASK_SIZE)
        strength = blob_df["Strength"].iloc[i]*0.3
        # add data. need to swap x and y
        if tx>0: 
            for ii in range(-3, 4):
                for jj in range(-3, 4):
                    if((ty+ii)<MASK_SIZE and (tx+jj<MASK_SIZE)):
                        mask[ty+ii,tx+jj,0] = strength
    smask_array.append(mask)

amask_array=[]
for counter in range(0, 1400):
    xy = labels[counter]
    xy_int = xy.astype(int)
    centerXY = xy_int[0:2]
    amask = np.zeros((MASK_SIZE, MASK_SIZE), dtype = "float")
    # add echo data
    if centerXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                amask[centerXY[1]+i,centerXY[0]+j] = xy[2]
    # add valve angle data
    vxy = valvelabels[counter]
    vxy_int = vxy.astype(int)
    vcenterXY = vxy_int[0:2]
    if vcenterXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                amask[vcenterXY[1]+i,vcenterXY[0]+j] = vxy[2]
    amask_array.append(amask)


val_smasks = np.stack(smask_array)
print(val_smasks.shape)

val_amasks = np.stack(amask_array)
print(val_amasks.shape)

# loading trainning data (8160 datasets or 8200 datasets)
imgnames = []
masknames = []
for counter in range(0, 9600):
    imgnames.append(f"./Images/{counter:03d}.jpg")
    masknames.append(f"./masks/{counter:03d}.png")

im_array = []
for i in tqdm(imgnames):
    im = cv2.imread(i)
    im = cv2.resize(im, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # need to convert image into float32 type before trainning
    gray = gray/255.0
    im_array.append(gray)

mask_array = []
for i in tqdm(masknames):
    im = cv2.imread(i)
    im = cv2.resize(im, dsize=(MASK_SIZE, MASK_SIZE))
    mask = im
    mask = mask/255.0
    mask_array.append(mask)
  
images = np.stack(im_array)
print(images.shape)

masks = np.stack(mask_array)
print(masks.shape)

# Create Size and Angle data
# Read label csv file
df = pd.read_csv('./labelsEcho9600.csv', delimiter=',')
df.drop(["imageId"], axis=1, inplace=True)
labels = df.to_numpy()
labels = [label * SCALE_VECTOR for label in labels]
labels = np.stack(labels)

# Read valve label csv file
valvedf = pd.read_csv('./labelsValve9600.csv', delimiter=',')
valvedf.drop(["imageId"], axis=1, inplace=True)
valvelabels = valvedf.to_numpy()
valvelabels = [label * SCALE_VECTOR for label in valvelabels]
valvelabels = np.stack(valvelabels)

# Read Blob label csv file
bdf = pd.read_csv('./blob9600.csv', delimiter=',')

smask_array = []
for counter in range(0, 9600):
    xy = labels[counter]
    xy_int = xy.astype(int)
    centerXY = xy_int[0:2]
    mask = np.zeros((MASK_SIZE, MASK_SIZE, 2), dtype = "float")
    # add echo data
    if centerXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                mask[centerXY[1]+i,centerXY[0]+j, 0] = xy[3]/MASK_SIZE
                mask[centerXY[1]+i,centerXY[0]+j, 1] = xy[4]/MASK_SIZE 
    # add valve size data
    vxy = valvelabels[counter]
    vxy_int = vxy.astype(int)
    vcenterXY = vxy_int[0:2]
    if vcenterXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                mask[vcenterXY[1]+i,vcenterXY[0]+j, 0] = vxy[3]/MASK_SIZE
                mask[vcenterXY[1]+i,vcenterXY[0]+j, 1] = vxy[4]/MASK_SIZE
    # Add blob data
    imgname=(f"{(counter):04d}.jpg")
    blob_df =bdf.loc[bdf['imageId'] == imgname]
    for i in range(len(blob_df)):
        tx = int(blob_df["Center X"].iloc[i] * MASK_SIZE)
        ty = int(blob_df["Center Y"].iloc[i] * MASK_SIZE)
        strength = blob_df["Strength"].iloc[i]*0.3
        # add data. need to swap x and y
        if tx>0: 
            for ii in range(-3, 4):
                for jj in range(-3, 4):
                    if((ty+ii)<MASK_SIZE and (tx+jj<MASK_SIZE)):
                        mask[ty+ii,tx+jj,0] = strength
    smask_array.append(mask)

amask_array=[]
for counter in range(0, 9600):
    xy = labels[counter]
    xy_int = xy.astype(int)
    centerXY = xy_int[0:2]
    amask = np.zeros((MASK_SIZE, MASK_SIZE), dtype = "float")
   # add echo data
    if centerXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                amask[centerXY[1]+i,centerXY[0]+j] = xy[2]
    # add valve size data
    vxy = valvelabels[counter]
    vxy_int = vxy.astype(int)
    vcenterXY = vxy_int[0:2]
    if vcenterXY[0]>0:
        for i in range(-3, 4):
            for j in range(-3, 4):
                amask[vcenterXY[1]+i,vcenterXY[0]+j] = vxy[2]
    amask_array.append(amask)

smasks = np.stack(smask_array)
print(smasks.shape)

amasks = np.stack(amask_array)
print(amasks.shape)


# Set Validation data
save_dir = "models"
model.validation_data = (val_images,[val_masks,val_smasks,val_amasks])
model.best_model_path = os.path.join(save_dir, "best_model_Three.hdf5")

# fit
metrics = DetMetrics()
history = model.fit(
    images,
    [masks,smasks,amasks],
    epochs= 50,
    batch_size=4,
    callbacks=[metrics],
)


