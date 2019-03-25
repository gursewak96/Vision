'''
    File name: monitor.py
    Author: Gursewak Singh
    Date Created: 24 January 2019
    Date last modified: 20 March 2019
    Python version: 2.7
    Status: Project
    Email: gursewak19d@gmail.com
'''
import math
import cv2
from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict
from ui import font

EAR_THRESH = 0.2  #Eye aspect ratio threshold
MAR_THRESH = 0.4   #Mouth aspect ration threshold

COUNTER_YAWN = 0
COUNTER_BLINK = 0

EYE_FRAMES = 0
MAR_FRAMES = 0

FRAMES_THRESH = 10

isActive = False

FACIAL_LANDMARKS_IDXS = OrderedDict([
                        ("mouth",(48,68)),
                        ("outer_mouth",(48,61)),
                        ("inner_mouth",(60,68)),
                        ("right_eyebrow",(17,22)),
                        ("left_eyebrow",(22,27)),
                        ("right_eye",(36,42)),
                        ("left_eye",(42,48)),
                        ("nose",(27,36)),
                        ("jaw",(0,17))
                        ])

(mStart,mEnd) = FACIAL_LANDMARKS_IDXS["outer_mouth"]
(leftEyeStart,leftEyeEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    """
    Takes the list of eye
    returns the aspect ratio of the given eye
    """
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    return (A+B)/(2.0*C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3],mouth[9])
    B = dist.euclidean(mouth[0],mouth[6])
    return A/B

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

def getEarAndMar(shape):
    """
    Assume the dlib: object
    Returns the mouth aspect ratio and eye aspect Ratio
    """
    shape = shape_to_np(shape)

    leftEye = shape[leftEyeStart:leftEyeEnd]
    rightEye = shape[rightEyeStart:rightEyeEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    mouth = shape[mStart:mEnd]

    return (leftEAR+rightEAR)/2.0,mouth_aspect_ratio(mouth)

def makeWindow(info,ear,mar):
    cv2.putText(info,"Driver Monitoring Display",(20,30),font,0.7,(255,255,255),2)
    cv2.line(info,(0,70),(512,70),(255,255,0))
    cv2.line(info,(0,75),(512,75),(255,255,0))
    cv2.putText(info,"Eye Aspect Ratio: {:.2f}".format(ear),(10,100),font,0.7,(255,255,0),1)
    cv2.putText(info, "Mouth Aspect Ratio: {:.2f}".format(mar), (10, 130),
        font, 0.7, (255, 255, 0), 1)
    cv2.putText(info, "Blinks: {:.2f}".format(COUNTER_BLINK), (10, 160),
        font, 0.7, (255, 255, 0), 1)
    cv2.putText(info, "Yawns: {}".format(COUNTER_YAWN), (10, 190),
        font, 0.7, (255, 255, 0), 1)
    cv2.line(info,(0,210),(512,210),(255,255,0))
    cv2.line(info,(0,215),(512,215),(255,255,0))
    cv2.putText(info,"Driver Status",(80,250),font,0.7,(255,255,255),2)
    if isActive :
        cv2.putText(info, "Active", (10, 280),
            font, 0.7, (0, 255, 0), 1)
    else:
        cv2.putText(info, "!Alert Driver is Drowsy!", (10, 280),
            font, 0.7, (0, 0, 255), 2)
    return info
