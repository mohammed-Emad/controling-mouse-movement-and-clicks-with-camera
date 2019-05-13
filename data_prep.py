import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util


dir = os.path.dirname(os.path.realpath(__file__))

TRAIN_DIR = dir + '/data/RIGHT_TRAIN'
TEST_DIR = dir + '/data/RIGHT_TEST'
IMG_SIZE = 100
LR = 1e-3 #learning rate
MODEL_NAME = 'left_eye-{}-{}.model'.format(LR, '6conv-basic-left')
R_MODEL_NAME = 'right_eye-{}-{}.model'.format(LR, '6conv-basic-left')

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [left ,negative]
    if word_label == 'negative': return [1, 0]
    elif word_label == 'right': return [0, 1]

def create_train_data():
    training_data = []
    training_data_rot = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img) #get the image name using first function
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) #convert to gray scale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
        random_degree = random.uniform(-25, 25)
        img_random_rot = sk.transform.rotate(img, random_degree)
        img_random_rot = cv2.resize(img_random_rot, (IMG_SIZE, IMG_SIZE))
        training_data_rot.append([np.array(img_random_rot), np.array(label)])
    data = training_data_rot + training_data
    shuffle(data)
    np.save('R_train_data.npy', data)
    return data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save(os.path.join(dir, 'R_test_data.npy'), testing_data)
    return testing_data

if __name__ == '__main__':
    train_data = create_train_data()

# If you have already created the dataset:
#train_data = np.load('train_data.npy')
