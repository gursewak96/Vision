'''
    File name: orient.py
    Author: Gursewak Singh
    Date Created: 15 February 2019
    Date last modified: 23 March 2019
    Python version: 2.7
    Status: Project
    Email: gursewak19d@gmail.com
'''

import numpy as np
import cv2
from ui import font
import math
#get the camera parameters
focal_length = 0
center_of_camera = [0,0]

#create camera matrix
camera_matrix = None

rotation_vector = None
translation_vector = None

model_vpoints = np.array([
                        (0.0,0.0,0.0),
                        (0.0,-330.0,-65.0),
                        (-225.0,170.0,-135.0),
                        (225.0,170.0,-135.0),
                        (-150.0,-150.0,-125.0),
                        (150.0,-150.0,-125.0)
                        ])

dist_coeffs = np.zeros((4,1))


center_Origin_vector = np.array([[50.0],
                                [50.0],
                                [50.0]])
def setParam(image):
    """
    Sets the parameters
    """
    size = image.shape
    global focal_length
    global center_of_camera
    global camera_matrix
    focal_length = size[1]
    center_of_camera = (size[1]/2,size[0]/2)
    camera_matrix = np.array([[focal_length,0,center_of_camera[0]],
                            [0,focal_length,center_of_camera[1]],
                            [0,0,1]], dtype = "double")
def getImagePoints(shape):
    image_vpoints = np.array([
                            (shape[30][0],shape[30][1]),
                            (shape[8][0],shape[8][1]),
                            (shape[36][0],shape[36][1]),
                            (shape[45][0],shape[45][1]),
                            (shape[48][0],shape[48][1]),
                            (shape[54][0],shape[54][1])
                            ], dtype='double')
    return image_vpoints

def getRotAndTransVector(image_vpoints):
    global rotation_vector
    global translation_vector
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_vpoints,image_vpoints,camera_matrix,dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
    return rotation_vector, translation_vector

def makeAxis(frame,origin):
    (nose_end_point2Dz,_) = cv2.projectPoints(np.array([(0.0,0.0,500.0)]),rotation_vector,translation_vector,camera_matrix,dist_coeffs)
    (nose_end_point2Dx,_) = cv2.projectPoints(np.array([(500.0,0.0,0.0)]),rotation_vector,translation_vector,camera_matrix,dist_coeffs)
    (nose_end_point2Dy,_) = cv2.projectPoints(np.array([(0.0,500.0,0.0)]),rotation_vector,translation_vector,camera_matrix,dist_coeffs)

    po = ( int(origin[0]), int(origin[1]))
    pz = ( int(nose_end_point2Dz[0][0][0]), int(nose_end_point2Dz[0][0][1]))
    px = ( int(nose_end_point2Dx[0][0][0]), int(nose_end_point2Dx[0][0][1]))
    py = ( int(nose_end_point2Dy[0][0][0]), int(nose_end_point2Dy[0][0][1]))

     #BGR
    cv2.line(frame, po, py,[0,255,0],4)
    cv2.putText(frame,"Y",py,font,0.7,(255,255,255),2)
    cv2.line(frame, po, px,[0,0,255],4)
    cv2.putText(frame,"X",px,font,0.7,(255,255,255),2)
    cv2.line(frame, po, pz,[255,0,0],4)
    cv2.putText(frame,"Z",pz,font,0.7,(255,255,255),2)

def makeWindow(frame,angles):
    pitch, yaw, roll = (angles[0]*57.230-360)%360,angles[1]*57.230,angles[2]*57.230
    cv2.putText(frame,"Head Rotation",(50,30),font,0.7,(255,255,255),2)
    cv2.line(frame,(0,40),(512,40),(255,255,0))
    cv2.line(frame,(0,45),(512,45),(255,255,0))
    cv2.putText(frame,"Pitch (in radian): {:.2f}".format(pitch),(10,70),font,0.7,(0,0,255),1)
    cv2.putText(frame, "Roll (in radian): {:.2f}".format(roll), (10, 100),
        font, 0.7, (255, 0, 0), 1)
    cv2.putText(frame, "Yaw (in radian): {:.2f}".format(yaw), (10, 130),
        font, 0.7, (0, 255, 0), 1)
    if(yaw >40 ):
        cv2.putText(frame, "Looking : Right", (10, 160),
            font, 0.7, (0, 255, 0), 1)
    elif(yaw< -25):
        cv2.putText(frame, "Looking : Left", (10, 160),
            font, 0.7, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "Looking : at the road", (10, 160),
            font, 0.7, (0, 255, 0), 1)
    cv2.line(frame,(0,175),(512,175),(255,255,0))
    cv2.line(frame,(0,170),(512,170),(255,255,0))
    cv2.putText(frame,"Head Position",(50,200),font,0.7,(255,255,255),2)
    cv2.line(frame,(0,210),(512,210),(255,255,0))
    cv2.line(frame,(0,215),(512,215),(255,255,0))
    cv2.putText(frame,"X : {:.2f}".format(translation_vector[0][0]),(10,240),font,0.7,(0,0,255),1)
    cv2.putText(frame,"Y : {:.2f}".format(translation_vector[1][0]),(10,270),font,0.7,(0,255,0),1)
    cv2.putText(frame,"Z : {:.2f}".format(translation_vector[2][0]),(10,300),font,0.7,(255,0,0),1)

def rotationMatrixToEulerAngles( R) :

        #assert(isRotationMatrix(R))

        #To prevent the Gimbal Lock it is possible to use
        #a threshold of 1e-6 for discrimination
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])
