from flask import Flask, render_template, Response
import numpy as np
import cv2
# import sys
# from driver import *
# import threading

# driver = Driver()
# driverThread = threading.Thread(target=driver.doSlam)
# driverThread.daemon = True
# driverThread.start()

# def get_map_frame():
#     frame = driver.slamAlgorithm.getMap()
#     ret, jpeg = cv2.imencode('.jpg', frame)
#     return jpeg.tobytes()

# def get_video_frame():
#     frame = driver.slamAlgorithm.matchviz
#     ret, jpeg = cv2.imencode('.jpg', frame)
#     return jpeg.tobytes()

app = Flask(__name__)
roverViewVideo = cv2.VideoCapture("rover-view.mp4")
humanViewVideo = cv2.VideoCapture("human-view.mp4")
mapVideo = cv2.VideoCapture("map.mp4")

def genRoverView():
    while True:
        success, image = roverViewVideo.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genHumanView():
    while True:
        success, image = humanViewVideo.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genMap():
    while True:
        success, image = mapVideo.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/outputs')
def outputs():
    return render_template('outputs.html', disabled=True)

@app.route('/roverView')
def roverView():
    return Response(genRoverView(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/humanView')
def humanView():
    return Response(genHumanView(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/map')
def map():
    return Response(genMap(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save')
def save():
    # driver.saveMap()
    return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000',debug=True)