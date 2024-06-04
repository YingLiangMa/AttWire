import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from skimage.feature import peak_local_max
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


model = generate_model()
model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), 
              loss={
                  'cen_out': FocalLoss,
                  'size_out': "mae",
                  'angle_out': "mae"})
model.summary()

# Load weights of the model
model.load_weights("./best_model_Three.hdf5")

for counter in range(0, 50):
    img = cv2.imread(f"./Images/{counter:03d}.jpg")
    # img = cv2.imread(f"./valimages/{counter:03d}.jpg")
    img = cv2.resize(img, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fimg = gray/ 255.0
    # reshape(1 (first one is the number of images, 200 (image width), 200 (image height), 1 (one color channel))
    in_img = np.array(fimg).reshape(1,IMAGE_SIZE,IMAGE_SIZE,1)
    outdata = model.predict(in_img)
    pred = np.squeeze(outdata[0])
    peaks = peak_local_max(pred[:,:,2], min_distance=1, threshold_rel=0.4) # threshold_rel is Minimum intensity of peaks.
    print(peaks)

    echocenmap = pred[:,:,0]
    # find the center of object
    indices = np.where(echocenmap == echocenmap.max())
    xx=indices[0]
    yy=indices[1]
    cx=xx[0]
    cy=yy[0]
    # print(cx,cy)
    print("echo", echocenmap[cx,cy])
    sizemap = np.squeeze(outdata[1])
    ww=sizemap[cx,cy,0]
    hh=sizemap[cx,cy,1]
    # print(ww,hh)
    anglemap = np.squeeze(outdata[2])
    angle = anglemap[cx,cy]
    # print(angle)
    xy = _make_box_pts(cy*2,cx*2,angle,ww*IMAGE_SIZE,hh*IMAGE_SIZE)
    xxyy = np.vstack((xy,xy[0,:]))

    fig, ax = plt.subplots()
    ax.plot(xxyy[:, 0], xxyy[:, 1], c="y")
    # Draw Rotated text
    dx = xxyy[2, 0] - xxyy[3, 0]
    dy = xxyy[2, 1] - xxyy[3, 1]
    rotn = np.degrees(np.arctan2(dy, dx))
    midx = (xxyy[2, 0] + xxyy[3, 0])/2
    midy = (xxyy[2, 1] + xxyy[3, 1])/2
    ax.text(midx, midy, '{0:.3f}'.format(echocenmap[cx,cy]), ha='center', va='bottom',color='yellow', fontsize=10,
        rotation=rotn, rotation_mode='anchor', transform_rotates_text=True)
    
    # find the center of Valve
    valvecenmap = pred[:,:,1]
    indices = np.where(valvecenmap == valvecenmap.max())
    xx=indices[0]
    yy=indices[1]
    cx=xx[0]
    cy=yy[0]
    # print(cx,cy)
    print("valve", valvecenmap[cx,cy])

    if valvecenmap[cx,cy]>0.5:
        ww=sizemap[cx,cy,0]
        hh=sizemap[cx,cy,1]
        angle = anglemap[cx,cy]
        xy = _make_box_pts(cy*2,cx*2,angle,ww*IMAGE_SIZE,hh*IMAGE_SIZE)
        xxyy = np.vstack((xy,xy[0,:]))
        ax.plot(xxyy[:, 0], xxyy[:, 1], c="b")
        # Draw Rotated text
        dx = xxyy[2, 0] - xxyy[3, 0]
        dy = xxyy[2, 1] - xxyy[3, 1]
        rotn = np.degrees(np.arctan2(dy, dx))
        midx = (xxyy[2, 0] + xxyy[3, 0])/2
        midy = (xxyy[2, 1] + xxyy[3, 1])/2
        ax.text(midx, midy, '{0:.3f}'.format(valvecenmap[cx,cy]), ha='center', va='bottom',color='blue', fontsize=10,
            rotation=rotn, rotation_mode='anchor', transform_rotates_text=True)
    
    # plt.imshow(pred[:,:,2])
    plt.imshow(img)
    # plt.plot(peaks[:, 1], peaks[:, 0], 'r+')
    plt.plot(peaks[:, 1]*2, peaks[:, 0]*2, 'r+')
    # plt.show()

    plt.show(block=False)
    plt.pause(1)
    plt.close()
