'''
    File name: main.py
    Author: Gursewak Singh
    Date Created: 20 January 2019
    Date last modified: 24 March 2019
    Python version: 2.7
    Status: Project
    Email: gursewak19d@gmail.com
'''

import cv2
import dlib
from collections import OrderedDict
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor/shape_predictor_68_face_landmarks.dat")


def shape_to_np(shape, dtype="int"):
    """
    Takes the shape dlib: full_object_detection and dtype
    Return the array of coordinates of landmarks
    """
    #initialise the list of x-y coordinates
    coord = np.zeros((shape.num_parts,2),dtype=dtype)

    #loop over facial landmarks and convert them into x,y tupeles
    for i in range(0,shape.num_parts):
        coord[i] = (shape.part(i).x,shape.part(i).y)

    return coord

def detect_face(gray):
    """
    return the face and the rectangle
    """
    #convert colored image to grayscale
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #to get the closest face
    x = 0
    y = 0
    w = 0
    h = 0
    max_area = 0

    faces = detector(gray)
    face = None
    rect = (x,y,w,h)
    for _face in faces:
        _x = _face.left()
        _y = _face.top()
        _w = _face.right() - _x
        _h = _face.bottom() - _y
        if _w*_h > max_area:
            max_area = _w*_h
            x = _x
            y = _y
            w = _w
            h = _h
            face = _face
            rect = x,y,w,h

    if face == None:
        return None,None

    else:
        return face,rect

def wireframe(image,face, color):
    shape = shape_to_np(face)

    for i in range(0,face.num_parts-1):
        pts = shape[i]
        cv2.circle(image,tuple(pts),2,color,-1)
