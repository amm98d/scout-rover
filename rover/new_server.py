import time
import zlib
import cv2 as cv
import numpy as np
from flask import Flask
from freenect import sync_get_depth as get_depth, sync_get_video as get_video

app = Flask(__name__)

st = -1
counter = 0


def convert_to_greyscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def gen_color():
    (frame, _) = get_video()
    # print(frame.shape, frame.dtype)
    frame = convert_to_greyscale(frame)
    # print(frame.shape)
    frame_bytes = frame.tobytes()
    # print(len(frame_bytes))
    compressed_frame_bytes = zlib.compress(frame_bytes, 1)
    # print(len(compressed_frame_bytes))
    return compressed_frame_bytes


def gen_depth():
    global st, counter
    (frame, _) = get_depth()
    # print(frame.shape, frame.dtype)
    counter += 1
    frame_bytes = frame.tobytes()
    # print(len(frame_bytes))
    compressed_frame_bytes = zlib.compress(frame_bytes, 1)
    # print(len(compressed_frame_bytes))
    print(counter / (time.time() - st))
    return compressed_frame_bytes


@app.route('/color')
def color():
    global st
    st = time.time() if st < 0 else st
    return gen_color()


@app.route('/rgb')
def rgb():
    return gen_depth()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

