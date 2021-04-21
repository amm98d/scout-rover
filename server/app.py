from flask import Flask, render_template, Response
import numpy as np
import cv2
import sys
from driver import *
import threading

driver = Driver()
driverThread = threading.Thread(target=driver.doSlam)
driverThread.start()

def get_map_frame():
    frame = driver.slamAlgorithm.map
    # frame= cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # encode OpenCV raw frame to jpg and displaying it
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def get_video_frame():
    frame = driver.slamAlgorithm.map
    # frame= cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # encode OpenCV raw frame to jpg and displaying it
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

app = Flask(__name__)

@app.route('/outputs')
def outputs():
    return render_template('outputs.html')

@app.route('/')
def index():
    return render_template('index.html')

def genMap():
    while True:
        frame = get_map_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genVideo():
    while True:
        frame = get_video_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/mapping')
def mapping():
    return Response(genMap(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(genVideo(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug=True)