# USAGE
# python 1_write_cats-vs-dogs_images.py
# (optional) -p /home/danieleb/ML/cats-vs-dogs/input/jpg

import numpy as np
import cv2
import os
import argparse
import glob
import sys
import shutil


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pathname", default="_dataset/train", help="Path to the dataset")
args = vars(ap.parse_args())
# root path name of dataset
path_root = args["pathname"]

if (not os.path.exists(path_root)):
    print('ERROR: you need the directory with the jpg files')
    sys.exit(0)


# Size of images
IMAGE_WIDTH  = 256 # 227
IMAGE_HEIGHT = 256 # 227

labelNames = ["cat", "dog", "others"]


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img


images_path = [img for img in glob.glob(path_root + "/*/*.jpg")]

# -----------------------------------------------------------------------------
print('BUILD THE VALIDATION SET with 4000 images: 2000 per each class')

wrk_dir = os.path.join(os.getcwd(), "_dataset", "_val")

if (os.path.exists(wrk_dir)):
    shutil.rmtree(wrk_dir)
os.mkdir(wrk_dir)

for s in labelNames[0:2]:
    path_name = os.path.join(wrk_dir, s)
    if (not os.path.exists(path_name)):
        os.mkdir(path_name)

with open(os.path.join(wrk_dir, "labels.txt"), "w") as f_lab:
    for s in labelNames:
        f_lab.write(s + "\n")

counter = [-1, -1, 0]
val_count = 0

f_test = open(os.path.join(wrk_dir, "validation.txt"), "w")
for in_idx, img_path in enumerate(images_path):
    # print("DBG: now processing image ", img_path)
    filename = os.path.basename(img_path)
    if '/cats/' in img_path:
        label = 0
    elif '/dogs/' in img_path:
        label = 1
    else: # other
        sys.exit(0)

    # skip the first 10500 images of each class and take the last 2000
    counter[ label ] = counter[ label ] + 1
    if (counter[ label ] <= 10499) :
        continue

    val_count = val_count + 1

    class_name = labelNames[label]
    string = " %1d" % label
    f_test.write(wrk_dir + "/" + class_name + "/" + filename + string + "\n")

    # Save resized image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    cv2.imwrite(
        os.path.join(wrk_dir, class_name, filename), 
        img.astype("int"))
f_test.close()

# -----------------------------------------------------------------------------
print('BUILD THE TEST SET with 1000 images of size 227 x 277')

wrk_dir = os.path.join(os.getcwd(), "_dataset", "test_images")

if (os.path.exists(wrk_dir)):
    shutil.rmtree(wrk_dir)
os.mkdir(wrk_dir)

with open(os.path.join(wrk_dir, "labels.txt"), "w") as f_lab:
    for s in labelNames:
        f_lab.write(s + "\n")

counter = [-1, -1, 0]
test_count = -1

f_test  = open(os.path.join(wrk_dir, "test.txt"), "w")
for in_idx, img_path in enumerate(images_path):
    filename = os.path.basename(img_path)
    if '/cats/' in img_path:
        label = 0
    elif '/dogs/' in img_path:
        label = 1
    else: # other
        sys.exit(0)

    # take the images from 10000 to 10500
    counter[ label ] = counter[ label ] + 1
    if (counter[ label ] <= 9999) or (counter[ label ] > 10499):
        continue

    test_count = test_count + 1

    string = " %04d" % test_count 
    f_test.write(wrk_dir + "/" + filename + string + "\n")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=227, img_height=227)
    cv2.imwrite(os.path.join(wrk_dir, filename), img.astype("int"))
f_test.close()

# -----------------------------------------------------------------------------
print('BUILD THE TRAIN IMAGES SET with 20000 images')

wrk_dir = os.path.join(os.getcwd(), "_dataset", "_train")

if (os.path.exists(wrk_dir)):
    shutil.rmtree(wrk_dir)
os.mkdir(wrk_dir)

for s in labelNames[0:2]:
    path_name = os.path.join(wrk_dir, s)
    if (not os.path.exists(path_name)):
        os.mkdir(path_name)

with open(os.path.join(wrk_dir, "labels.txt"), "w") as f_lab:
    for s in labelNames:
        f_lab.write(s + "\n")

counter = [-1,-1,0]
train_count = 0

f_test = open(os.path.join(wrk_dir, "train.txt"), "w")
for in_idx, img_path in enumerate(images_path):
    filename = os.path.basename(img_path)
    if '/cats/' in img_path:
        label = 0
    elif '/dogs/' in img_path:
        label = 1
    else: # other
        sys.exit(0)

    counter[ label ] = counter[ label ] + 1
    # skip images after the first 10000
    if (counter[ 0 ] > 9999) and (counter[ 1 ] > 9999):
        break

    train_count = train_count + 1

    class_name = labelNames[label]

    string = " %1d" % label
    f_test.write(
        wrk_dir + "/" + class_name + "/" + filename + string + "\n")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    cv2.imwrite(
        os.path.join(wrk_dir, class_name, filename),
        img.astype("int"))
f_test.close()

# -----------------------------------------------------------------------------
print('BUILD THE CALIBRATION IMAGES SET with 200 images')

wrk_dir = os.path.join(os.getcwd(), "_dataset", "_calib")

if (os.path.exists(wrk_dir)):
    shutil.rmtree(wrk_dir)
os.mkdir(wrk_dir)

for s in labelNames[0:2]:
    path_name = os.path.join(wrk_dir, s)
    if (not os.path.exists(path_name)):
        os.mkdir(path_name)

counter = [-1, -1, 0]
calib_count = -1

f_calib = open(os.path.join(wrk_dir, "calibration.txt"), "w")
for in_idx, img_path in enumerate(images_path):
    filename = os.path.basename(img_path)
    if '/cats/' in img_path:
        label = 0
    elif '/dogs/' in img_path:
        label = 1
    else: # other
        sys.exit(0)

    # take only the first 100 images per each class
    counter[ label ] = counter[ label ] + 1
    if (counter[ label ] > 99) : 
        continue

    calib_count = calib_count + 1

    class_name = labelNames[ label ]

    string2 = " %1d" % int(calib_count)
    f_calib.write(class_name + "/" + filename + string2 + "\n")

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    cv2.imwrite(
        os.path.join(wrk_dir, class_name, filename), 
        img.astype("int"))
f_calib.close()


print("Train      set contains ", train_count, " images")
print("Validation set contains ", val_count,   " images")
print("Calibrationset contains ", calib_count + 1, " images")
print("END\n")
