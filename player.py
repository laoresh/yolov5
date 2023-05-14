import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
cap = cv2.VideoCapture(r'runs\detect\exp3\20220113T141400Z_20220113T141900Z.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, [1280, 640])
    cv2.imshow('image', frame)
    k = cv2.waitKey(20)
    #q键退出
    if (k & 0xff == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
# from PIL import Image
# import os
# from os import path
# absolute_path = r'D:\yolov5-5.0\dataset_bcs\images\train'
# dirname = os.listdir(absolute_path)
# for i in dirname:
#     pathname = absolute_path + '/' + i
#     try:
#         img = Image.open(pathname)
#     except IOError:
#         print(pathname)
#     try:
#         img = np.asarray(img)
#     except:
#         print('corrupt img', pathname)
