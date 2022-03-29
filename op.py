import cv2
from cv2 import imshow
from cv2 import waitKey
import numpy as np

img = cv2.imread("matt2.png")

# First we crop the sub-rect from the image
x, y, w, h = 100, 100, 200, 100
sub_img = img[y:y+h, x:x+w]
white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

# Putting the image back to its position
img[y:y+h, x:x+w] = res

cv2.imshow('op',img)
waitKey()