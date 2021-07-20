#!/usr/bin/python3
from statistics import mean
import tensorflow as tf
import numpy as np
import cv2
import wmi


model = tf.keras.models.load_model('fd_model')  # loading trained model

"""------------------prepare_image--------------------------------
usage: used to modify frame according to model requirements
image: a single frame
return value: modified frame
"""


def prepare_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = np.array(img).reshape(1, 28, 28, 1)
    img = tf.cast(img, tf.float32)
    return img


cap = cv2.VideoCapture(0)  # start video recording

frame_count = 0  # number of recorded frames
frame_results = []  # list of frame values
while True:
    _, frame = cap.read()  # get a single frame
    frame_count += 1
    result = round(float(model.predict(prepare_image(frame))[0]), 4)  # feed frame to model
    if frame_count > 20:
        avg = mean(frame_results)
        frame_results = []
        frame_count = 0
        if avg > 0.6:
            # raise screen brightness to maximum
            brightness = 100
            c = wmi.WMI(namespace='wmi')
            methods = c.WmiMonitorBrightnessMethods()[0]
            methods.WmiSetBrightness(brightness, 0)
        else:
            # decrease screen brightness to 10%
            brightness = 10
            c = wmi.WMI(namespace='wmi')
            methods = c.WmiMonitorBrightnessMethods()[0]
            methods.WmiSetBrightness(brightness, 0)
    else:
        frame_results.append(result)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()  # stop video recording
