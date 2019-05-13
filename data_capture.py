from multiprocessing import Process, Manager, Value
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

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.int32([vertices]), 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def primaryFunction(x_dir, y_dir):
	#shape_predictor_68_face_landmarks.dat file must be located in same dir as this file
	#get current directory
    dir = os.path.dirname(os.path.realpath(__file__))
    print("Current project directory: " + dir)
    data_path = dir + "/data/RIGHT_TRAIN/"

	#initialize dlib's face detector and load prediticor data file
    print("Loading facial landmark prediction data files...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dir + "/shape_predictor_68_face_landmarks.dat")

	#initialize the video stream
    print("Camera sensor is getting started...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    image_counter = 10000

	#loop over the frames from the video stream
    while True:
		#grab frames from video stream, resize frame and converto to grayscale
        frame = vs.read()
        time.sleep(0.03)

		#crop frame and convert to grayscale
        (frame_height, frame_width, c) = frame.shape
        upper_left = (int(frame_height*.2), int(frame_width*.2))
        bottom_right = (int(frame_height - frame_height*.2), int(frame_width - frame_width*.2))
        frame = frame[upper_left[0] : bottom_right[0], upper_left[1] : bottom_right[1]]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#detect faces in the grayscale frame
        detections = detector(gray, 0)
        cv2.imshow("original", frame)

        if len(detections) > 0:
			#loop over each detected face
            for face in detections:
                (X, Y, W, H) = face_utils.rect_to_bb(face)
                if X > 0 and Y > 0 and H > 0 and W > 0:
                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    # mid_left = [shape[27,0],shape[22,1]]
                    # lower_left = [shape[54,0],shape[33,1]]
                    # vertices = np.array([[shape[16]],[shape[26]],[shape[24]],[shape[22]],[mid_left],[shape[29]],[lower_left],[shape[13]],[shape[14]],[shape[15]],[shape[16]]])
                    mid_right = [shape[27,0],shape[21,1]]
                    lower_right = [shape[49,0],shape[35,1]]
                    vertices_right = np.array([[shape[0]],[shape[17]],[shape[19]],[shape[21]],[mid_right],[shape[29]],[lower_right],[shape[3]],[shape[2]],[shape[1]],[shape[0]]])
                    frame_masked = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_masked = roi(frame_masked, vertices_right)
                    # eye_vertices = np.array([[shape[45]],[shape[44]],[shape[43]],[shape[42]],[shape[47]],[shape[46]],[shape[45]]])
                    eye_vertices = np.array([[shape[36]],[shape[37]],[shape[38]],[shape[39]],[shape[40]],[shape[41]],[shape[36]]])
                    frame_masked = cv2.fillPoly(frame_masked, np.int32([eye_vertices]), 0)
                    frame_face = imutils.resize(frame_masked[Y : (Y + H), X : (X + W)], width=200)
                    frame_face = cv2.Canny(frame_face, 40, 40)
                    cv2.imshow("masked", frame_face)

                    image_name = "right.{}.png".format(image_counter)
                    cv2.imwrite(os.path.join(data_path, image_name), frame_face)
                    image_counter += 1
                    print(image_counter)

                    #LEFT_EYE MODEL
                    # img = cv2.resize(frame_face, (50, 50))
                    # img = adjust_gamma(img)
                    # cv2.imshow("augmented data", img)


        key = cv2.waitKey(1) & 0xFF

		#kill loop on key press
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
	x_dir = Value('d', 0.0)
	y_dir = Value('d', 0.0)
	p1 = Process(target=primaryFunction, args=(x_dir, y_dir))
	p1.start()
	p1.join()
