import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
from helpers import score_iou, _make_box_pts

import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout,concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Attention,
    Conv2DTranspose,
    Flatten,
    Dense,
)

IMAGE_SIZE = 304
MASK_SIZE = 152


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


# Define loss functions
def reg_loss(y_true, y_pred):
    yaw_true, w_true, h_true = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    yaw_pred, w_pred, h_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    loss = tf.abs(w_true - w_pred) + tf.abs(h_true - h_pred) + 2*np.pi*tf.abs(yaw_true-yaw_pred)
    return loss

def FocalLoss(y_true, y_pred):    
    beta = 0.8
    gamma = 2

    # inputs = K.flatten(y_true)
    # targets = K.flatten(y_pred)
    f_loss = beta * (1 - y_pred) ** gamma * y_true * K.log(y_pred)  # β*(1-p̂)ᵞ*p*log(p̂)
    f_loss += (1 - beta) * y_pred ** gamma * (1 - y_true) * K.log(1 - y_pred)  # (1-β)*p̂ᵞ*(1−p)*log(1−p̂)
    f_loss = -f_loss  # −[β*(1-p̂)ᵞ*p*log(p̂) + (1-β)*p̂ᵞ*(1−p)*log(1−p̂)]

    # Average over each data point/image in batch
    axis_to_reduce = range(1, K.ndim(f_loss))
    f_loss = K.mean(f_loss, axis=axis_to_reduce)
    
    return f_loss

def FocalValue(y_true, y_pred):
    return -FocalLoss(y_true, y_pred)


model = generate_model()
model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), 
              loss={
                  'cen_out': FocalLoss,
                  'size_out': "mae",
                  'angle_out': "mae"})
model.summary()

# Load weights of the model
model.load_weights("./best_model_Three.hdf5")

SaveFolder = "./output/"

for counter in range(0, 50):
    # img = cv2.imread(f"./valimages/{counter:03d}.jpg")
    img = cv2.imread(f"./Images/{counter:03d}.jpg")
    img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fimg = gray/ 255.0
    # reshape(1 (first one is the number of images, 200 (image width), 200 (image height), 1 (one color channel))
    in_img = np.array(fimg).reshape(1,IMAGE_SIZE,IMAGE_SIZE,1)
    outdata = model.predict(in_img)
    cenmap = np.squeeze(outdata[0])
    echocenmap = cenmap[:,:,0] 
    # find the center of Valve
    valvecenmap = cenmap[:,:,1]
    blobcenmap = cenmap[:,:,2]

    # Draw center heatmap on the image
    EchoMap = cv2.resize(echocenmap, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    EchoMapInt = np.uint8((1.0-EchoMap)*255)
    echo_color = cv2.applyColorMap(EchoMapInt,  cv2.COLORMAP_JET)
    for x in range(0,IMAGE_SIZE):
        for y in range(0,IMAGE_SIZE):
            channels_xy = EchoMap[y,x]
            if channels_xy >0.2:
                img[y,x] = 0.5*echo_color[y,x] + 0.5*img[y,x]

    # Draw center heatmap on the image
    ValveMap = cv2.resize(valvecenmap, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    ValveMapInt = np.uint8((1.0-ValveMap)*255)
    valve_color = cv2.applyColorMap(ValveMapInt,  cv2.COLORMAP_JET)
    for x in range(0,IMAGE_SIZE):
        for y in range(0,IMAGE_SIZE):
            channels_xy = ValveMap[y,x]
            if channels_xy >0.2:
                img[y,x] = 0.5*valve_color[y,x] + 0.5*img[y,x]

    # Draw catheter electrode center heatmap on the image
    BlobMap = cv2.resize(blobcenmap, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    BlobMapInt = np.uint8((1.0-BlobMap)*255)
    blob_color = cv2.applyColorMap(BlobMapInt,  cv2.COLORMAP_JET)
    for x in range(0,IMAGE_SIZE):
        for y in range(0,IMAGE_SIZE):
            channels_xy = BlobMap[y,x]
            if channels_xy >0.2:
                img[y,x] = 0.5*blob_color[y,x] + 0.5*img[y,x]

    cv2.imwrite(SaveFolder+f"{counter:03d}.tif", img)
    plt.imshow(img)
    # plt.savefig(SaveFolder+f"pos{counter:03d}.jpg")
    plt.show(block=False)
    plt.pause(1)
    plt.close()
