from scipy import rand
from sklearn import utils
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
import skimage.io as io
import skimage as sk
from skimage.transform import resize
import os
import csv
import random
from configurators import train_config
import utils as util
import matplotlib.pyplot as plt

def random_binary(p):
    if(random.random()<p):
        return 1
    else:
        return 0


def get_ae_model(t_conf):

    input_img = Input(shape=(t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3))  
    x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
    x = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 
    x = Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 
    x = Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    # The decoding process
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='Adam', loss='mse')
    return autoencoder

def generator(dataframe, x_path, config):
    IMG_SIZE = config.IMG_WIDTH
    batch_size = config.BATCH_SIZE
    c = 0
    dataframe = dataframe.sample(frac=1)  #Shuffle dataset using sample method
    while (True):
        x_batch = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3))
        y_batch = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3))
        #sample_weight = np.zeros((batch_size, IMG_SIZE, IMG_SIZE))

        for i in range(c, c+batch_size):
            x = io.imread(x_path+dataframe.loc[i,'Image'])
            x = sk.resize(x, (IMG_SIZE, IMG_SIZE))
            x = image.img_to_array(x)/255.
            x_batch[i-c], y_batch[i-c] = x,x

        c+=batch_size
        if(c+batch_size>len(dataframe.index)):
            c=0
            dataframe = dataframe.sample(frac=1)
        yield x_batch, y_batch



def write_to_csv():
    #read images and labels from dataset and convert to csv file (1.)
    a = ['Images', 'Subsets']#'Subsets'
    list = os.listdir('C:/Users/Asus/Desktop/Codes/datasets/APTOS/dataset_15&19/')#Disease
    for i in range(len(list)):
        if(random_binary(0.1)):
            row = [list[i], 'Validation']
        else:
            row = [list[i], 'Training']
        
        with open('ae_dataset.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)  

def try_samples(model, t_conf):
    d_path = 'C:/Users/Asus/Desktop/Codes/datasets/APTOS/dataset_15&19/'
    im_list = os.listdir(d_path)#Disease
    for im in im_list:
        x = io.imread(d_path+im)
        img_in = resize(x, (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH))
        #img_in = image.img_to_array(x)/255.
        pred = model.predict(np.reshape(img_in, (1,t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3)))
        out_im = np.reshape(pred, (t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3))
        util.show_images([img_in,out_im])



def plot_loss_and_acc(history, save=False, saveDir='out/', fname=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    loss = history['loss']
    val_loss = history['val_loss']

    acc = history['accuracy']
    val_acc = history['val_accuracy']

    ax1.plot(loss, label='Train loss')
    ax1.plot(val_loss, label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(acc, label='Train accuracy')
    ax2.plot(val_acc, label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    if(save):
        plt.savefig(saveDir + fname + 'loss_and_acc.png')
    plt.show(block=False)
    plt.close()

if (__name__ == "__main__"):
    ae_model = util.load_model("auotencoder1")
    t_conf = train_config()
    try_samples(ae_model, t_conf)
