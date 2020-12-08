import sys
sys.path.append("../common/")
from NumpySocket import NumpySocket

import time
import numpy as np
from freenect import sync_get_depth as get_depth, sync_get_video as get_video

host_ip = '192.168.100.14'

npSocket = NumpySocket()
npSocket.startServer(host_ip, 9999)

while True:
	frame, _ = get_video()
	print(len(frame),len(frame[0]),len(frame[0][0]))
	time.sleep(0.016)
	npSocket.sendNumpy(frame)

try:
    npSocket.endServer()
except OSError as err:
    print("error")
