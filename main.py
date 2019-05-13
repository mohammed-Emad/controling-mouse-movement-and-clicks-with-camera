#REFERENCES/SOURCES http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
#https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
#https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/

from multiprocessing import Process, Manager, Value
import multiprocessing
from threading import Thread, Lock
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import imutils
import time
import dlib
import cv2
import win32api
import sys
import math
import numpy as np
from scipy.spatial import distance as dist
import os
from directkeys import SetPosition, moveMouseRel, PressKey, ReleaseKey, W, A, S, D, left_click_down, left_click_up, right_click_down, right_click_up
import win32api
import win32gui
import win32con
from data_prep import LR, IMG_SIZE, MODEL_NAME, R_MODEL_NAME
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
model = tflearn.DNN(convnet, tensorboard_dir='log')
model.load(MODEL_NAME)
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
model_R = tflearn.DNN(convnet, tensorboard_dir='log')
model_R.load(R_MODEL_NAME)

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.int32([vertices]), 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def setTopMost(wname):
        while True:
            hwndCap = win32gui.FindWindow(None, wname)
            if hwndCap != 0:
                win32gui.SetWindowPos(hwndCap, win32con.HWND_TOPMOST, 50, 50, 0, 0, 1)

def primaryFunction(x_dir, y_dir, l_prob, r_prob):
    #shape_predictor_68_face_landmarks.dat file must be located in same dir as this file
    #get current directory
    dir = os.path.dirname(os.path.realpath(__file__))
    print("Current project directory: " + dir)

    #initialize dlib's face detector and load prediticor data file
    print("Loading facial landmark prediction data files...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dir + "/shape_predictor_68_face_landmarks.dat")

    #initialize the video stream
    print("Camera sensor is getting started...")
    vs = VideoStream(src=0).start()
    #vs = cv2.VideoCapture(0)
    time.sleep(3.0)

    #loop over the frames from the video stream
    while True:
		#grab frames from video stream, resize frame and converto to grayscale
        # time.sleep(0.001)
        frame = vs.read()

        #crop frame and convert to grayscale
        (frame_height, frame_width, c) = frame.shape
        upper_left = (int(frame_height*.2), int(frame_width*.2))
        bottom_right = (int(frame_height - frame_height*.2), int(frame_width - frame_width*.2))
        frame = frame[upper_left[0] : bottom_right[0], upper_left[1] : bottom_right[1]]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detect faces in the grayscale frame
        detections = detector(gray, 0)

        if len(detections) > 0:
            #loop over each detected face
            for face in detections:
                (X, Y, W, H) = face_utils.rect_to_bb(face)
                if X > 0 and Y > 0 and H > 0 and W > 0:
                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    #LEFT DATA PREP
                    mid_left = [shape[27,0],shape[22,1]]
                    lower_left = [shape[54,0],shape[33,1]]
                    vertices = np.array([[shape[16]],[shape[26]],[shape[24]],[shape[22]],[mid_left],[shape[29]],[lower_left],[shape[13]],[shape[14]],[shape[15]],[shape[16]]])
                    frame_masked = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_masked = roi(frame_masked, vertices)
                    eye_vertices = np.array([[shape[45]],[shape[44]],[shape[43]],[shape[42]],[shape[47]],[shape[46]],[shape[45]]])
                    frame_masked = cv2.fillPoly(frame_masked, np.int32([eye_vertices]), 0)
                    frame_face = imutils.resize(frame_masked[Y : (Y + H), X : (X + W)], width=200)
                    frame_face = cv2.Canny(frame_face, 40, 40)
                    cv2.imshow("augemented left", frame_face)

                    mid_right = [shape[27,0],shape[21,1]]
                    lower_right = [shape[49,0],shape[35,1]]
                    vertices_right = np.array([[shape[0]],[shape[17]],[shape[19]],[shape[21]],[mid_right],[shape[29]],[lower_right],[shape[3]],[shape[2]],[shape[1]],[shape[0]]])
                    R_frame_masked = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    R_frame_masked = roi(R_frame_masked, vertices_right)
                    #eye_vertices = np.array([[shape[45]],[shape[44]],[shape[43]],[shape[42]],[shape[47]],[shape[46]],[shape[45]]])
                    R_eye_vertices = np.array([[shape[36]],[shape[37]],[shape[38]],[shape[39]],[shape[40]],[shape[41]],[shape[36]]])
                    R_frame_masked = cv2.fillPoly(R_frame_masked, np.int32([R_eye_vertices]), 0)
                    R_frame_face = imutils.resize(R_frame_masked[Y : (Y + H), X : (X + W)], width=200)
                    R_frame_face = cv2.Canny(R_frame_face, 40, 40)
                    cv2.imshow("masked", R_frame_face)

                    #LEFT_EYE MODEL
                    img = cv2.resize(frame_face, (IMG_SIZE, IMG_SIZE))
                    testing_data = np.array(img)
                    testing_data = testing_data.reshape(-1, 100, 100, 1).astype("float")
                    model_out = model.predict(testing_data)
                    l_prob.value = model_out[0,0]
                    #print(model_out[0,0])
                    cv2.putText(frame, str(model_out[0,0]), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1)

                    #RIGHT_EYE MODEL
                    R_img = cv2.resize(R_frame_face, (IMG_SIZE, IMG_SIZE))
                    R_testing_data = np.array(R_img)
                    R_testing_data = R_testing_data.reshape(-1, 100, 100, 1).astype("float")
                    R_model_out = model_R.predict(R_testing_data)
                    r_prob.value = R_model_out[0,0]
                    cv2.putText(frame, str(R_model_out[0,0]), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1)

				###################### MOUSE MOVEMENT CONTROL #####################
                # x_A = shape[0,0]
                # y_A = shape[0,1]
                x_A = (shape[36,0] + shape[0,0])/2
                y_A = (shape[36,1] + shape[0,1])/2
                #cv2.circle(frame, (x_A, y_A), 4, (255, 0, 0), 1)
                # x_B = shape[16,0]
                # y_B = shape[16,1]
                x_B = (shape[45,0] + shape[16,0])/2
                y_B = (shape[45,1] + shape[16,1])/2
                #cv2.circle(frame, (x_B, y_B), 4, (255, 0, 0), 1)
                x_C = shape[51,0]
                y_C = shape[51,1]
                #cv2.circle(frame, (x_C, y_C), 4, (255, 0, 0), 1)
                #cv2.line(frame, (x_A, y_A), (x_B, y_B), (0, 255, 0), 1)
                #cv2.line(frame, (x_A, y_A), (x_C, y_C), (0, 255, 0), 1)
                #cv2.line(frame, (x_B, y_B), (x_C, y_C), (0, 255, 0), 1)

                y_slope = y_B-y_A
                x_slope = x_B-x_A
                nonzero = 0.000000000000001

                #Calculate the x and y value for the point D
                x_Dnumerator = (y_C + float(x_slope)/float(y_slope + nonzero) * x_C) - (y_A - float(y_slope)/float(x_slope + nonzero) * x_A)
                x_Ddenominator = float(y_slope)/float(x_slope + nonzero) + float(x_slope)/float(y_slope + nonzero)
                x_D = float(x_Dnumerator/x_Ddenominator)

                y_Dnumerator = (y_C + float(x_slope)/float(y_slope + nonzero) * x_C) * (float(y_slope)/float(x_B-x_A + nonzero)) + (y_A - float(y_slope)/float(x_slope + nonzero) * x_A) * (float(x_slope)/float(y_slope + nonzero))
                y_Ddenominator = float(x_slope)/float(y_slope + nonzero) + float(y_slope)/float(x_slope + nonzero)
                y_D = y_Dnumerator/y_Ddenominator

                #cv2.circle(frame, (int(x_D), int(y_D)), 4, (255, 0, 0), 1)
                #cv2.line(frame, (int(x_D), int(y_D)), (x_C, y_C), (0, 255, 0), 1)

                AD = dist.euclidean((x_A, y_A), (x_D, y_D))
                DB = dist.euclidean((x_D, y_D), (x_B, y_B))
                DC = dist.euclidean((x_D, y_D), (x_C, y_C))
                AC = dist.euclidean((x_A, y_A), (x_C, y_C))
                CB = dist.euclidean((x_C, y_C), (x_B, y_B))

                theta_1 = math.acos((math.pow(DC, 2) - math.pow(AC, 2) - math.pow(AD, 2))/(-2*AC*AD + nonzero))
                theta_2 = math.acos((math.pow(AD, 2) - math.pow(AC, 2) - math.pow(DC, 2))/(-2*AC*DC + nonzero))*.75
                theta_3 = math.acos((math.pow(DB, 2) - math.pow(CB, 2) - math.pow(DC, 2))/(-2*CB*DC + nonzero))*.75
                theta_4 = math.acos((math.pow(DC, 2) - math.pow(CB, 2) - math.pow(DB, 2))/(-2*CB*DB + nonzero))

                xdir_ratio = (DB - AD + nonzero) / (AD + DB + nonzero)
                ydir_ratio = ((theta_1 + theta_4 + nonzero) - (theta_2 + theta_3 + nonzero)) / ((theta_1 + theta_4 + nonzero) + (theta_2 + theta_3 + nonzero))


                # x_dir.value = ((0.75*math.pow(xdir_ratio, 3) + xdir_ratio) * 10)#f(x) = (x^3 + x) * 10
                # y_dir.value = (0.75*(math.pow(ydir_ratio, 3) + ydir_ratio) * 10) + 2.5
                #if -0.3 < xdir_ratio < 0:
                x_dir.value = (4*xdir_ratio*math.pow(math.tan(xdir_ratio), 2) + .2*xdir_ratio) * 30
                y_dir.value = (4*ydir_ratio*math.pow(math.tan(ydir_ratio), 2) + .2*ydir_ratio) * 60 - 1

        #show frame
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        #kill loop on key press
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

def inputFunction(x_dir, y_dir, l_prob, r_prob):
    left_close_counter = 0
    left_open_counter = 1
    right_close_counter = 0
    right_open_counter = 1

    while True:
        #MOUSE MOVEMENT
        x_speed_limit = 50
        y_speed_limit = 20

        if x_dir.value > x_speed_limit:
            x_val = x_speed_limit
        elif x_dir.value < -1*x_speed_limit:
            x_val = -1*x_speed_limit
        else:
            x_val = x_dir.value

        if y_dir.value > y_speed_limit:
            y_val = y_speed_limit
        elif y_dir.value < -1*y_speed_limit:
            y_val = -1*y_speed_limit
        else:
            y_val = y_dir.value

        moveMouseRel(int(x_val), int(y_val))

        #LEFT MOUSE CLICK LOGIC
        if l_prob.value < 0.01:
            left_close_counter += 1
        if left_close_counter == 1:
            left_click_down()
            left_open_counter = 0
        if l_prob.value > 0.98:
            left_open_counter += 1
        if left_open_counter == 1:
            left_click_up()
            left_close_counter = 0

        #RIGHT MOUSE CLICK LOGIC
        if r_prob.value < 0.01:
            right_close_counter += 1
        if right_close_counter == 1:
            right_click_down()
            right_open_counter = 0
        if r_prob.value > 0.98:
            right_open_counter += 1
        if right_open_counter == 1:
            right_click_up()
            right_close_counter = 0

        time.sleep(0.01666)
        # time.sleep(0.03333)

if __name__ == '__main__':
    x_dir = Value('d', 0.0)
    y_dir = Value('d', 0.0)
    l_prob = Value('d', 1.0)
    r_prob = Value('d', 1.0)
    p1 = Process(target=primaryFunction, args=(x_dir, y_dir, l_prob, r_prob))
    p1.start()
    p2 = Process(target=inputFunction, args=(x_dir, y_dir, l_prob, r_prob))
    p2.start()
    p1.join()
    p2.join()
