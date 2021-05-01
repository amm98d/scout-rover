import numpy as np
import cv2 as cv2
import urllib.request
import io
import zlib

def uncompress_nparr(bytestring):
    """
    """
    return np.load(io.BytesIO(zlib.decompress(bytestring)))

while(True):
    byte_array = urllib.request.urlopen('http://192.168.100.113:5000/video_feed').read()
    frame = uncompress_nparr(byte_array)
    # frame = np.frombuffer(uncompressed_bytearray, dtype='uint8').reshape((480, 640, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
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