import cv2
import numpy as np
import augmentations

from augment_and_mix import augment_and_mix


if __name__ == '__main__':
    img = cv2.imread('test_img.jpg')
    img = cv2.resize(img, (augmentations.IMAGE_SIZE, augmentations.IMAGE_SIZE))
    img = np.float32(img)
    mixed_img = augment_and_mix(img)

    cv2.imwrite('mixed.jpg', mixed_img)
