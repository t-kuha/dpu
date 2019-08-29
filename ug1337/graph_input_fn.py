import os
import cv2


calib_image_dir = "_calib_dir"
flist = os.listdir(calib_image_dir)
calib_batch_size = 50

num_calib_img = len(flist)

def calib_input(iter):
  images = []
  for b in range(0, calib_batch_size):
    idx = iter * calib_batch_size + b
    if idx >= num_calib_img:
      idx = idx % len(flist)

    calib_image_name = flist[idx]

    # read image as grayscale, returns numpy array (28,28)
    image = cv2.imread(
      os.path.join(calib_image_dir, calib_image_name), cv2.IMREAD_GRAYSCALE)

    # scale the pixel values to range 0 to 1.0
    image = image/255.0

    # reshape numpy array to be (28,28,1)
    image = image.reshape((image.shape[0], image.shape[1], 1))
    images.append(image)
  return {"images_in": images}
