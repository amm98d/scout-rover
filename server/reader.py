# import numpy as np

# depth = np.zeros(shape=(480, 640)).flatten()
# rgb = np.zeros(shape=(480, 640, 3)).flatten()
# print(depth.shape)
# print(rgb.shape)

# stacked = np.hstack((depth,rgb))
# print(stacked.shape)

# nDepth = stacked[:307200].reshape(480,640)
# nRgb = stacked[307200:1228800].reshape(480,640,3)
# print(nDepth.shape)
# print(nRgb.shape)

import numpy as np
import cv2 as cv2
import urllib.request
import io
import zlib

def uncompress_nparr(bytestring):
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

count=1
while(True):
    print(count)
    bytes = urllib.request.urlopen('http://192.168.100.113:5000/kinect_feed').read()
    stacked_arr = uncompress_nparr(bytes)
    depth = stacked_arr[:307200].reshape(480,640)
    rgb = stacked_arr[307200:1228800].reshape(480,640,3)
    cv2.imshow('depth',depth)
    cv2.imshow('rgb',rgb)
    cv2.imwrite(f'saves/depth{count}.png',depth)
    cv2.imwrite(f'saves/rgb{count}.png',rgb)
    count+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()