import cv2
import numpy as np
import os

def getFacePosition(image):
    """list형식의 x, y, w, h의 반환값으로 좌표와 넓이를 나타냄"""
    faceCascade= cv2.CascadeClassifier(r'face_detecting_data\haarcascade_frontalface_alt.xml')
    img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(255,0,0),2)

    return faces

def getFaceRect(image, faces):

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    return image