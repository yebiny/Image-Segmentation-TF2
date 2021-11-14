import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import utils
from PIL import ImageOps
import matplotlib.pyplot as plt



class DataGenerator():
    def __init__(self, img_size, batch_size, ds_name='human'):
        self.img_size = img_size
        self.batch_size = batch_size
        self.ds_name = ds_name

    def read_img_from_path(self, path, mask=False):
        img = tf.io.read_file(path)
        if mask:
            img = tf.image.decode_png(img, channels=1)
            #img.set_shape([None, None, 1], dtype=tf.uint8)
            img = tf.image.resize(images=img,
                                  size=[self.img_size,self.img_size])
            img = tf.compat.v1.to_int32(img)
            if self.ds_name=='pet':
                img = img-1
        else:
            img = tf.image.decode_png(img, channels=3)
            #img.set_shape([None, None, 3])
            img = tf.image.resize(images=img, size=[self.img_size,self.img_size])
            img = img /127.5 -1

        return img

    def load_dataset(self, img_path, mask_path):
        img = self.read_img_from_path(img_path)
        mask = self.read_img_from_path(mask_path, mask=True)
        return img, mask

    def generate(self, img_paths, mask_paths):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        dataset = dataset.map(self.load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

def display_img(img_path, mask_path, mask_cmap=None):

    if type(img_path)==str:
      img = utils.load_img(img_path)
      print('input size: ', img.size)
    else:
      img = img_path

    if type(mask_path)==str:
      mask = utils.load_img(mask_path, color_mode='grayscale')
      mask = ImageOps.autocontrast(mask)
      print('mask size:', mask.size)
    else:
      mask = mask_path


    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(mask, alpha=0.6, cmap=mask_cmap)

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.imshow(mask, cmap=mask_cmap)

    plt.show()
    
def main():

    dg = DataGenerator(IMG_SIZE, BATCH_SIZE)
    
if __name__=='__main__':
    main()
