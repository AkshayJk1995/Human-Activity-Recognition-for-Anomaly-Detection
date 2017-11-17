# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 10:05:13 2016

@author: AkshayJk
"""

import cv2

cap = cv2.VideoCapture('E:/Final Year Project/Datasets/ICPR/test/59_20_2.avi')
while not cap.isOpened():
    cap = cv2.VideoCapture('E:/Final Year Project/Datasets/ICPR/test/59_20_2.avi')
    cv2.waitKey(1000)
    print("Wait for the header")

pos_frame = cap.get(1)
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        #cv2.imshow('video', frame)
        cv2.imshow("image", frame);
        cv2.waitKey(30);
        pos_frame = cap.get(1)
        pos_frame+=13461
        cv2.imwrite("E:/Final Year Project/Datasets/ICPR/test/%d.jpg" % pos_frame,frame)
        print(str(pos_frame)+" frames")
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(1, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(1) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break