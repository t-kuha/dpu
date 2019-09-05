import os
import shutil
import cv2

import tensorflow as tf


CALIB_DIR = os.path.join(os.getcwd(), '_calib_dir')
TESTIMG_DIR = os.path.join(os.getcwd(), '_cifar10_test')


if (os.path.exists(CALIB_DIR)):
    shutil.rmtree(CALIB_DIR)
os.makedirs(CALIB_DIR)

if (os.path.exists(TESTIMG_DIR)):
    shutil.rmtree(TESTIMG_DIR)
os.makedirs(TESTIMG_DIR)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# create file for list of calibration images
for i in range(len(x_test)):
    cv2.imwrite(os.path.join(CALIB_DIR, 'calib_' + str(i) + '.png'), x_test[i])

for i in range(len(x_test)):
    cv2.imwrite(os.path.join(TESTIMG_DIR, str(i) + '.png'), x_test[i])

print ('FINISHED GENERATING CALIBRATION IMAGES')
