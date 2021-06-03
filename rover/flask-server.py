from freenect import sync_get_depth as get_depth, sync_get_video as get_video
from flask import Flask, render_template, make_response
import io
import zlib
import numpy as np

def compress_nparr(nparr):
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed, len(uncompressed), len(compressed)

app = Flask(__name__)

@app.route('/kinect_feed')
def kinect_feed():
    print("RECEIVED")
    (depth,_), (rgb,_) = get_depth(), get_video()
    print(depth.flatten().shape)
    print(rgb.flatten().shape)
    stacked_arr = np.hstack((depth.flatten(),rgb.flatten()))
    print(stacked_arr.shape)
    bytes = compress_nparr(stacked_arr)[0]
    response = make_response(bytes)
    response.headers.set('Content-Type', 'application/octet-stream')
    return response

@app.route('/rgb')
def rgb():
    print("RGB RECEIVED")
    (rgb,_) = get_video()
    rgb_bytes = compress_nparr(rgb)[0]
    response = make_response(rgb_bytes)
    response.headers.set('Content-Type', 'application/octet-stream')
    return response

@app.route('/depth')
def depth():
    print("DEPTH RECEIVED")
    (depth,_) = get_depth()
    depth_bytes = compress_nparr(depth)[0]
    response = make_response(depth_bytes)
    response.headers.set('Content-Type', 'application/octet-stream')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')
