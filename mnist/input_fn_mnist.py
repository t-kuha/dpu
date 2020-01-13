import cv2
import os
import glob

calib_image_dir = '_mnist_calib'
calib_batch_size = 50

img_list = sorted(glob.glob(os.path.join(calib_image_dir, '*.png')))

def calib_input(iter):
    images = []

    print('iter = %d' % iter)
    
    # line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        idx_ = (iter * calib_batch_size + index) % len(img_list)
        path = img_list[idx_]

        # read image as grayscale, returns numpy array (28,28)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # scale the pixel values to range 0 to 1.0
        image = image/255.0

        # reshape numpy array to be (28,28,1)
        image = image.reshape((image.shape[0], image.shape[1], 1))
        images.append(image)

    return {"conv2d_input": images}
