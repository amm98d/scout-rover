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

@app.route('/video_feed')
def video_feed():
    (_,_), (rgb,_) = get_depth(), get_video()
    rgb_bytes = compress_nparr(rgb)[0]
    response = make_response(rgb_bytes)
    response.headers.set('Content-Type', 'application/octet-stream')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000')
