# from server.server import Server
from flask import Flask, render_template, Response
import cv2
# import numpy as np
# import sys
from server import *
import threading

server = Server()
serverThread = threading.Thread(target=server.startIt)
serverThread.daemon = True
# serverThread.start()

humanViewVideo = cv2.VideoCapture("http://192.168.100.45:8080/video")

app = Flask(__name__)

# def genRoverView():
#     while True:
#         frame = server.slamAlgorithm.video_frame
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genHumanView():
    while True:
        success, image = humanViewVideo.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# def genMap():
#     while True:
#         frame = server.slamAlgorithm.map
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/outputs')
def outputs():
    return render_template('outputs.html', disabled=True)

# @app.route('/roverView')
# def roverView():
#     return Response(genRoverView(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/humanView')
def humanView():
    return Response(genHumanView(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/map')
# def map():
#     return Response(genMap(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/save')
# def save():
#     # driver.saveMap()
#     return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug=True)