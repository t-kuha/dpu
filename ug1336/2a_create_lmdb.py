# ##################################################################################################
# USAGE
# python 2a_create_lmdb.py -i /home/ML/cats-vs-dogs/input/jpg/ -o /home/ML/cats-vs-dogs/input/lmdb

# it reads the CIFAR10 JPG images and creates 2 LMDB databases:
# train_lmdb (20000 images in LMDB) and  val_lmdb (4000 images in LMDB) to be used during the training

# ##################################################################################################

import os
import sys
import shutil
import glob
import random
import numpy as np
import cv2
import argparse

import caffe
from caffe.proto import caffe_pb2
import lmdb


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inp", default = "_dataset", help="path to the input jpeg dir")
ap.add_argument("-o", "--out", default = "_lmdb", help="path to the output lmdb dir")
args = vars(ap.parse_args())

# this is the directory where input JPG images are placed
IMG_DIR = args["inp"]   
# this is the directory where lmdb will be placed
WORK_DIR= args["out"]   

# Size of images
IMAGE_WIDTH  = 256 #227
IMAGE_HEIGHT = 256 #227

# create "WORK_DIR" directory if it does not exist
if (os.path.exists(WORK_DIR)): 
    shutil.rmtree(WORK_DIR)
os.mkdir(WORK_DIR)

train_lmdb = os.path.join(WORK_DIR, 'train_lmdb')
valid_lmdb = os.path.join(WORK_DIR, 'valid_lmdb')

if (not os.path.exists(train_lmdb)):
    os.mkdir(train_lmdb)

if (not os.path.exists(valid_lmdb)):
    os.mkdir(valid_lmdb)

# -----------------------------------------------------------------------------
def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

# -----------------------------------------------------------------------------
train_data = [img for img in glob.glob(os.path.join(IMG_DIR, "_train/*/*.jpg"))]
valid_data = [img for img in glob.glob(os.path.join(IMG_DIR, "_val/*/*.jpg"))]

# -----------------------------------------------------------------------------
print('Creating train_lmdb')

# Shuffle train_data
random.seed(48)
random.shuffle(train_data)

num_train_images = 0
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        num_train_images = 1 + num_train_images

        filename = os.path.basename(img_path)
        if 'cat' in filename:
            label = 0
        elif 'dog' in filename:
            label = 1
        else:
            print("ERROR: there is a class which is not part of the dataset")
            sys.exit(0)
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
        # '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

# -----------------------------------------------------------------------------
# Create the validation LMDB
print('Creating valid_lmdb')
random.seed(48)
random.shuffle(valid_data)

num_valid_images = 0
in_db = lmdb.open(valid_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(valid_data):
        num_valid_images = 1 + num_valid_images

        filename = os.path.basename(img_path)
        if 'cat' in filename:
            label = 0
        elif 'dog' in filename:
            label = 1
        else:
            print("ERROR: there is a class which is not part of the dataset")
            sys.exit(0)        

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
        # '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print('Finished processing all images')
print(' Number of images in LMDB training   dataset ',  num_train_images)
print(' Number of images in LMDB validation dataset ',  num_valid_images)
