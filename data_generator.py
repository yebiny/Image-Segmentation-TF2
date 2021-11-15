import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy

class DataGenerator():
    def __init__(self, img_size, batch_size, ds_name='human'):
        self.img_size = img_size
        self.batch_size = batch_size
        self.ds_name = ds_name

    def path2arr(self, path, mask=False):
        if not mask: 
          arr = load_img(path, target_size=(self.img_size, self.img_size), color_mode='rgb')
          arr = img_to_array(arr, dtype='uint8')
        else:
          arr = load_img(path, target_size=(self.img_size, self.img_size), color_mode='grayscale')
          arr = img_to_array(arr, dtype='uint8')
        
        return arr

    def load_dataset(self, img_paths, mask_paths):
        imgs = []
        masks = []
        for img_path, mask_path in zip(img_paths, mask_paths):
          img = self.path2arr(img_path)
          mask = self.path2arr(mask_path, mask=True)
          imgs.append(img)
          masks.append(mask)
        return np.array(imgs), np.array(masks)

    def preprocess(self, img, mask):

      if self.ds_name=='pet':
        img = (tf.cast(img, tf.float32) / 127.5) -1
        mask = tf.cast(mask, tf.float32)-1
      
      elif self.ds_name=='human':
        img = (tf.cast(img, tf.float32) / 127.5) -1
        mask = tf.cast(mask, tf.float32)
      
      else:
        print("dataset name error")

      return img, mask
    
    def augment(self, img, mask):

        img = tf.image.random_brightness(img, max_delta=0.5)
        img = tf.clip_by_value(img, -1, 1)

        ds = tf.concat((img, tf.cast(mask, tf.float32)), axis=2)
        crop_size = int(self.img_size/4)
        ds = tf.image.resize_with_crop_or_pad(ds, self.img_size + crop_size, self.img_size + crop_size) 
        ds = tf.image.random_crop(ds, size=[self.img_size, self.img_size, 4])
        img, mask = ds[:,:,:3], ds[:,:,-1]
        mask = tf.expand_dims(mask, axis=2)
        return img, mask

    def generate(self, img_paths, mask_paths, aug=False):
        imgs, masks = self.load_dataset(img_paths, mask_paths)
        dataset = tf.data.Dataset.from_tensor_slices((imgs, masks))
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if aug:
          dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset
