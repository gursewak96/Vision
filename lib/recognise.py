'''
    File name: recognise.py
    Author: Gursewak Singh
    Date Created: 27 February 2019
    Date last modified: 19 March 2019
    Python version: 2.7
    Status: Project
    Email: gursewak19d@gmail.com
'''

import cv2
import dlib
import numpy as np
import os
from common import detect_face
from ui import font
import os

subjects = ["","Gursewak Singh","Gourav Sharma","Raveena"]
CONFIDENCE_THRESH = 50
faces = []
labels = []
face_recognizer = None

def prepareTrainingData(dataFolderPath):
    global faces, labels
    dirs = os.listdir(dataFolderPath)
    for dir in dirs:
        if not dir.startswith("s"):
            continue
        label = int(dir.replace("s",""))

        subPath = dataFolderPath+"/"+dir

        subjectImages = os.listdir(subPath)
        for subjectImage in subjectImages:
            path = subPath+"/"+subjectImage
            img = cv2.imread(path)
            _,rect= detect_face(img)
            if rect != None:
                face = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                faces.append(face)
                labels.append(label)
    return faces, labels


def predict(test_image,rect):
    x,y,w,h = rect
    face = test_image[y:y+h,x:x+w]
    label,confidence = face_recognizer.predict(face)
    if confidence < CONFIDENCE_THRESH:
        label_text = subjects[label]
    else:
        label_text = "unknown"
    return label_text,confidence

def prepareRecogniser():
    global face_recognizer
    #prepare the faces and labels
    print("preparing training data")
    faces, labels = prepareTrainingData("recognizerData/trainingData")
    print("Data prepared")
    print("Total faces: ",len(faces))
    print("Total labels: ",len(labels))

    #get the instance of face recognizer
    face_recognizer = cv2.face.createLBPHFaceRecognizer()
    face_recognizer.train(faces,np.array(labels))

def makeWindow(text,confidence,driverFrame,frame,rect):
    cv2.putText(driverFrame,"Driver Details",(50,30),font,0.7,(255,255,255),2)
    cv2.line(driverFrame,(0,40),(512,40),(255,255,0))
    cv2.line(driverFrame,(0,45),(512,45),(255,255,0))
    if text == "unknown":
        cv2.putText(driverFrame,"Name: "+text,(10,100),font,0.7,(0,0,255),1)
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255),3)

    else:
        cv2.putText(driverFrame,"Name: "+text,(10,100),font,0.7,(0,255,0),1)
        cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),3)

    cv2.putText(driverFrame, "Confidence : {:.2f}".format(confidence), (10, 160),
    font, 0.7, (255, 255, 0),1)

def trainRecognizer(frame,set):
    path = "recognizerData/trainingData/"+set
    dirs = os.listdir(path)
    cv2.imwrite(path+"/"+str(len(dirs)+1)+".jpg",frame)
