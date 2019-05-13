import os

dir = os.path.dirname(os.path.realpath(__file__))

path = dir + '/data/RIGHT_TRAIN'
files = os.listdir(path)
i = 1021

for file in files:
    if file.startswith("neg"):
        os.rename(os.path.join(path, file), os.path.join(path, 'left.' + str(i)+'.jpg'))
    i = i+1
