import numpy as np
import cv2
import utils



img0 = np.zeros((171, 224), np.uint8)
img0[:, :] = 255,

img1 = np.zeros((171, 224), np.uint8)
img1[:, :] = 255,

img0 = cv2.resize(img0, (0, 0), fx=1.5, fy=1.5)
img1 = cv2.resize(img1, (0, 0), fx=1.5, fy=1.5)

utils.draw_demo_ui("g00", img0=img0, img1=img1)

cv2.waitKey(0)