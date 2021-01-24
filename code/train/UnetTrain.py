import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
from skimage.io import imread
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
TRAINOUT_PATH = r"D:\SEgmentation\data2\train\train_images_256"
MASKOUT_PATH = r"D:\SEgmentation\data2\train\train_masks_256"
TEST_PATH = r"D:\SEgmentation\data2\test\train_images_256"
MASKTEST_PATH = r"D:\SEgmentation\data2\test\train_masks_256"

seed = 42
random.seed = seed
np.random.seed = seed

#initialise arrays
X_train = np.zeros((len(os.listdir(TRAINOUT_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(os.listdir(TRAINOUT_PATH)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
X_test = np.zeros((len(os.listdir(TEST_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(os.listdir(MASKTEST_PATH)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

#'Read train , test images and masks ... '#
n = 0
for filename in os.listdir(TRAINOUT_PATH):
    if filename.endswith(".tif"):
        img = imread(TRAINOUT_PATH +"\\" + filename)
        img = resize(img/256, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = np.expand_dims(img, axis=-1)
        X_train[n] = img
        mask_ = imread(MASKOUT_PATH +"\\" +  filename)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        Y_train[n] = mask_
        n = n+1
n = 0
for filename in os.listdir(TEST_PATH):
    if filename.endswith(".tif"):
        img = imread(TEST_PATH +"\\" + filename)
        img = resize(img/256, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = np.expand_dims(img, axis=-1)
        X_test[n] = img
        mask_ = imread(MASKTEST_PATH +"\\" +  filename)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        Y_test[n] = mask_
        n = n+1

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        m = tf.keras.metrics.MeanIoU(num_classes=2)
        preds = (y_pred > 0.5).astype(np.uint8)
        y_truevalue = y_true.astype(np.uint8)
        m.update_state(y_truevalue, preds)
        x = m.result().numpy()
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
first = Lambda(lambda x: x / 255) (inputs)

con1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (first)
con1 = Dropout(0.1) (con1)
con1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con1)
p1 = MaxPooling2D((2, 2)) (con1)

con2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
con2  = Dropout(0.1) (con2)
con2  = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con2)
p2 = MaxPooling2D((2, 2)) (con2)

con3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
con3  = Dropout(0.2) (con3 )
con3  = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con3)
p3 = MaxPooling2D((2, 2)) (con3)

con4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
con4 = Dropout(0.2) (con4)
con4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con4)
p4 = MaxPooling2D(pool_size=(2, 2)) (con4)

con5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
con5 = Dropout(0.3) (con5)
con5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (con5)
u6 = concatenate([u6, con4])
con6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
con6 = Dropout(0.2) (con6)
con6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (con6)
u7 = concatenate([u7, con3])
con7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
con7= Dropout(0.2) (con7)
con7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (con7)
u8 = concatenate([u8, con2])
con8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
con8 = Dropout(0.1) (con8)
con8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (con8)
u9 = concatenate([u9, con1], axis=3)
con9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
con9 = Dropout(0.1) (con9)
con9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (con9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (con9)

model = Model(inputs=[inputs], outputs=[outputs])
metrics = ["acc", iou]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics= metrics)
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=100,callbacks=[earlystopper, checkpointer])
model.save("model.h5")

#Predict on train and test
m = tf.keras.metrics.MeanIoU(num_classes=2)
preds_train = model.predict(X_train, verbose=1)
preds_train = (preds_train > 0.5).astype(np.uint8)
Y_traint = Y_train.astype(np.uint8)
m.update_state(Y_traint, preds_train)
print("validation iou {}".format(m.result().numpy()))

preds_train = model.predict(X_test, verbose=1)
preds_train = (preds_train > 0.5).astype(np.uint8)
Y_testt = Y_test.astype(np.uint8)
m.update_state(Y_testt, preds_train)
print("Test iou {}".format(m.result().numpy()))





