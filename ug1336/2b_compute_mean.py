# -----------------------------------------------------------------------------
# USAGE
# python 2b_compute_mean.py

# It reads the images from LMDB training database and create the mean file
# -----------------------------------------------------------------------------

import os
import numpy as np
import caffe

# -----------------------------------------------------------------------------
# working directories
INP_LMDB = os.path.join("_lmdb", "train_lmdb")
MEAN_FILE = os.path.join("_lmdb", "mean.binaryproto")

# -----------------------------------------------------------------------------
# MEAN of all training dataset images
print ('Generating mean image of all training data')
mean_command = "compute_image_mean -backend=lmdb "
os.system(mean_command + INP_LMDB + ' ' + MEAN_FILE)

# -----------------------------------------------------------------------------
# show the mean image
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(MEAN_FILE, 'rb').read()
blob.ParseFromString(data)

mean_array = np.asarray(
    blob.data, 
    dtype=np.float32).reshape((blob.channels, blob.height, blob.width))
print(" mean value channel 0: ", np.mean(mean_array[0,:,:]))
print(" mean value channel 1: ", np.mean(mean_array[1,:,:]))
print(" mean value channel 2: ", np.mean(mean_array[2,:,:]))

# -----------------------------------------------------------------------------
# THE RESULT SHOULD BE SOMETHING LIKE THIS IN CASE OF HISTOGRAM EQUALIZATION:
# I0123 14:21:34.758246 84929 compute_image_mean.cpp:114] Number of channels: 3
# I0123 14:21:34.758345 84929 compute_image_mean.cpp:119] mean_value channel [0]: 106.409
# I0123 14:21:34.758442 84929 compute_image_mean.cpp:119] mean_value channel [1]: 116.049
# I0123 14:21:34.758523 84929 compute_image_mean.cpp:119] mean_value channel [2]: 124.467
