import sys
sys.path.append("../common/")
from NumpySocket import NumpySocket

import numpy as np
import cv2

npSocket = NumpySocket()
npSocket.startClient(9999)

while True:
	frame = npSocket.recieveNumpy()
	print(len(frame))
	#cv2.imshow("The display window" , frame)

try:
    npSocket.endServer()
except OSError as err:
    print("error",err)
