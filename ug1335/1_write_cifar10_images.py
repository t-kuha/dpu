# USAGE
# python 1_write_cifar10_images.py


import numpy as np
from tensorflow.keras.datasets import cifar10
import cv2
import os


# create "path_root" directory if it does not exist
path_root = "_cifar10_jpg"  
if (not os.path.exists(path_root)): 
    os.mkdir(path_root)


# load the training and testing data from CIFAR-10 dataset
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY ), (testX, testY )) = cifar10.load_data()


# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# BUILD THE TEST SET with 1000 images
# create "test" directory if it does not exist
wrk_dir = os.path.join(path_root, "test")
if (not os.path.exists(wrk_dir)): 
    os.mkdir(wrk_dir)

f_lab  = open(os.path.join(wrk_dir, "labels.txt"), "w")
for s in range(0, len(labelNames)):
    string = "%s\n" % labelNames[s] 
    f_lab.write(string) 
f_lab.close()
    
f_test  = open(os.path.join(wrk_dir, "test.txt"), "w")
f_test2 = open(os.path.join(wrk_dir, "test2.txt"), "w")

counter = [0,0,0,0,0,0,0,0,0,0]

test_count = 0
for i in range(0, len(testY)):
    idx = int(testY[i])
    counter[ idx ] = counter[ idx ] + 1

    # take only the first 100 images per each class
    if counter[ idx ] > 100: 
        continue

    string = "%05d" % counter[ idx ]

    class_name = labelNames[ idx ]
    
    path_name = os.path.join(wrk_dir, class_name + "_" + string + ".jpg")

    string2 = " %1d" % test_count
    f_test.write(path_name + string2 + "\n")
    f_test2.write(class_name + "_" + string + ".jpg" + string2 + "\n")

    cv2.imwrite(
        os.path.join(wrk_dir, class_name + "_" + string + ".jpg"),
        testX[i])

    test_count = test_count + 1

f_test.close()
f_test2.close()


# BUILD THE CALIBRATION IMAGES SET
wrk_dir = os.path.join(path_root, "calib")
if (not os.path.exists(wrk_dir)):
    os.mkdir(wrk_dir)

f_calib = open(os.path.join(wrk_dir, "calibration.txt"), "w")   

counter = [0,0,0,0,0,0,0,0,0,0]
calib_count = 0
for i in range(0, len(trainY)) : 
    idx = int(trainY[i])
    counter[ idx ] = counter[ idx ] + 1

    # take only the first 100 images per each class
    if counter[ idx ] > 100 :
        continue

    string = "%05d" % counter[ idx ]

    class_name = labelNames[ idx ]

    dir_name = os.path.join(wrk_dir, class_name)
    if (not os.path.exists(dir_name)):
        os.mkdir(dir_name)

    # path_name = os.path.join(wrk_dir, class_name, class_name + "_" + string + ".jpg")

    string2 = " %1d" % calib_count 
    f_calib.write(class_name + "/" + class_name + "_" + string + ".jpg" + string2 + "\n")

    cv2.imwrite(
        os.path.join(wrk_dir, class_name, class_name + "_" + string + ".jpg"),
        trainX[i])
    
    calib_count = calib_count + 1

f_calib.close()

print("Test        set contains ", test_count,  " images")
print("Calibration set contains ", calib_count, " images")
