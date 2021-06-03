# import numpy as np

# depth = np.zeros(shape=(480, 640))
# rgb = np.zeros(shape=(480, 640, 3))
# print(depth.shape)
# print(rgb.shape)

# print((np.vstack((depth,rgb))).shape)
# print(np.array().shape)

import numpy as np
import cv2 as cv2
import urllib.request
import io
import zlib

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

count=1
while(True):
    print(count)
    rgb_byte_array = urllib.request.urlopen('http://192.168.100.113:5000/rgb').read()
    depth_byte_array = urllib.request.urlopen('http://192.168.100.113:5000/depth').read()
    rgb = uncompress_nparr(rgb_byte_array)
    depth = uncompress_nparr(depth_byte_array)
    # frame = np.frombuffer(uncompressed_bytearray, dtype='uint8').reshape((480, 640, 3))
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('rgb',rgb)
    cv2.imshow('depth',depth)
    # cv2.imwrite(f'saves/rgb/rgb{count}.png',rgb)
    # cv2.imwrite(f'saves/depth/depth{count}.png',depth)
    count+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# while True:
#     imgResp = urllib.request.urlopen('http://192.168.100.113:5000/video_feed')
#     # print(type(imgResp))
#     x = imgResp.read()
#     # print(type(x))
#     # print(len(x))
#     y = np.frombuffer(x, dtype='uint8').reshape((480, 640, 3))
#     # print(y.shape)
#     # cv2.imwrite('color_img.jpg', y)
#     cv.imshow("image", y)
#     # cv.waitKey()