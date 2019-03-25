'''
    File name: ui.py
    Author: Gursewak Singh
    Date Created: 18 January 2019
    Date last modified: 24 March 2019
    Python version: 2.7
    Status: Project
    Email: gursewak19d@gmail.com
'''

import cv2
import numpy as np

# frames for user interface
#driverDet #to show the driver detail
menu = np.zeros((412,312,3), np.uint8)     #to show the menu
monDet = np.zeros((312,312,3),np.uint8)  #to show the monitoring detail
#posDet    #to show the position detail
#help      #to show the help window to the user
font = cv2.FONT_HERSHEY_SIMPLEX

def makeMenu():
    global menu
    cv2.putText(menu,"Application Menu",(60,30),font,0.7,(255,255,255),2)
    cv2.line(menu,(0,50),(312,50),(255,255,0))
    cv2.line(menu,(0,54),(312,54),(255,255,0))
    cv2.putText(menu,"m: to start the Monitoring system.",(10,90),font,0.5,(0,255,255))
    cv2.putText(menu,"h: to start the Head Position system.",(10,120),font,0.5,(0,255,255))
    cv2.putText(menu,"d: to show the driver information.",(10,150),font,0.5,(0,255,255))
    cv2.putText(menu,"w: to show the skeleton.",(10,180),font,0.5,(0,255,255))
    cv2.putText(menu,"q: to quit the application.",(10,210),font,0.5,(0,255,255))

def splash():
    cv2.namedWindow("Monitoring System")
    cv2.moveWindow("Monitoring System",400,200)
    splash = cv2.imread("media/splash.png")
    cv2.imshow("Monitoring System",splash)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
