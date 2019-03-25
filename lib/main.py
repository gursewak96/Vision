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
import time
import ui
import common
import numpy as np
import monitor
import orient
import recognise
import argparse

#creating cammand line interface
ap = argparse.ArgumentParser()
ap.add_argument("-c","--camera",required=False,help="Camera number from where to get feed.",default=0)
args = vars(ap.parse_args())
print(args)

#initialise recognizer module
recognise.prepareRecogniser()

#create splash for one second
ui.splash()


#modules flags
SHOW_MONITORING_SYS = False
SHOW_HEAD_POSE_3D = False
SHOW_DRIVER_DETAILS = False
SHOW_WIREFRAME = False
Grab = False

#create two primary Windows
cv2.namedWindow("Menu")
cv2.moveWindow("Menu",50,50)
cv2.namedWindow("Live Feed")
cv2.moveWindow("Live Feed",370,50)

#initialise frames
menu = ui.makeMenu()

#time variable
currentTime = time.time()

#initialise the camera and get the first frame and sets the param for the orient
cap = cv2.VideoCapture(0)
_,frame = cap.read()
orient.setParam(frame)

#get and produce the frame on every loop
while True:
    #get the feed from the camera
    _,frame = cap.read()

    #detect the face in the frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    (face,rect) = common.detect_face(gray)

    #if face is detected
    if (face != None):
        #get the points on the face
        shape = common.predictor(gray,face)
        if Grab:
            recognise.trainRecognizer(frame,"s1")
            Grab = not Grab

        #logic to perform face monitoring
        if SHOW_MONITORING_SYS :
            infoFrame = np.zeros((312,312,3),np.uint8)
            #get ear and mar
            ear,mar = monitor.getEarAndMar(shape)

            if ear < monitor.EAR_THRESH:
                monitor.EYE_FRAMES += 1

                if monitor.EYE_FRAMES >= monitor.FRAMES_THRESH:
                    monitor.isActive = False
            else:
                monitor.isActive = True
                if monitor.EYE_FRAMES > 2:
                    monitor.COUNTER_BLINK += 1
                monitor.EYE_FRAMES = 0

            if mar >= monitor.MAR_THRESH:
                monitor.MAR_FRAMES += 1
            else:
                if monitor.MAR_FRAMES > monitor.FRAMES_THRESH:
                    monitor.COUNTER_YAWN += 1
                monitor.MAR_FRAMES = 0

            infoFrame = monitor.makeWindow(infoFrame,ear,mar)
            cv2.namedWindow("Monitor")
            cv2.moveWindow("Monitor",1020,50)
            cv2.imshow("Monitor",infoFrame)

        #logic to calculate position Details
        if SHOW_HEAD_POSE_3D :
            positionFrame = np.zeros((330,312,3),np.uint8)
            imagePointsVector = orient.getImagePoints(monitor.shape_to_np(shape))
            rotation_vector,translation_vector = orient.getRotAndTransVector(imagePointsVector)
            origin = int(imagePointsVector[0][0]), int(imagePointsVector[0][1])
            orient.makeAxis(frame,origin)

            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles = orient.rotationMatrixToEulerAngles(rmat)
            orient.makeWindow(positionFrame,angles)
            cv2.namedWindow('Head Pose')
            cv2.moveWindow('Head Pose',1020,400)
            cv2.imshow('Head Pose',positionFrame)

        #logic for face recognition
        if SHOW_DRIVER_DETAILS :
            #create driver frame
            driverFrame = np.zeros((212,312,3),np.uint8)

            #detect the person
            text,confidence = recognise.predict(gray,rect)
            recognise.makeWindow(text,confidence,driverFrame,frame,rect)
            cv2.namedWindow("Driver Details")
            cv2.moveWindow('Driver Details',50,500)
            cv2.imshow('Driver Details',driverFrame)

        #logic to perform wireframe overlay
        if SHOW_WIREFRAME :
            mark = common.predictor(gray,face)
            frame = np.zeros(frame.shape)
            common.wireframe(frame,mark,(255,255,255))


    cv2.imshow("Live Feed",frame)
    cv2.imshow("Menu",ui.menu)

    #response entered by the user
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('m'):
        SHOW_MONITORING_SYS = not SHOW_MONITORING_SYS
    if key == ord('h'):
        SHOW_HEAD_POSE_3D = not SHOW_HEAD_POSE_3D
    if key == ord('d'):
        SHOW_DRIVER_DETAILS = not SHOW_DRIVER_DETAILS
    if key == ord('w'):
        SHOW_WIREFRAME = not SHOW_WIREFRAME
    if key == ord('g'):
        Grab = not Grab


#destroy the resources
cap.release()
cv2.destroyAllWindows()
