from flask import Flask, render_template, Response
import numpy as np
import cv2
import sys
from driver import *
import threading

driver = Driver()
driverThread = threading.Thread(target=driver.doSlam)

def get_map_frame():
    frame = driver.slamAlgorithm.getMap()
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def get_video_frame():
    frame = driver.slamAlgorithm.matches
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

app = Flask(__name__)

def genMap():
    while True:
        frame = get_map_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genVideo():
    while True:
        frame = get_video_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/outputs')
def outputs():
    driverThread.start()
    return render_template('outputs.html')

@app.route('/mapping')
def mapping():
    return Response(genMap(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(genVideo(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug=True)