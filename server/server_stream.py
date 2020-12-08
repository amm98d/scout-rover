#########Server##########################
# import the necessary packages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2

# initialize the ImageHub object
imageHub = imagezmq.ImageHub()
# start looping over all the frames
i=0
while True:
	# receive RPi name and frame from the RPi and acknowledge
	# the receipt
    (rpiName, frame) = imageHub.recv_image()
    #cv2.imwrite("received" + str(i) +'.jpg' , frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("The display window" , frame)
    cv2.waitKey(1)
    #cv2.waitKey(0);
    #cv2.destroyAllWindows();
    #cv2.waitKey(1)
    imageHub.send_reply(b'OK')
    i+=1






