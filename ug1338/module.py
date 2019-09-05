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

    # read image, returns numpy array
    image = cv2.imread(
      os.path.join(calib_image_dir, calib_image_name))

    # scale the pixel values to range 0 to 1.0
    image = image/255.0

    # reshape numpy array
    image = image.reshape((image.shape[0], image.shape[1], 3))
    images.append(image)
  return {"images_in": images}
