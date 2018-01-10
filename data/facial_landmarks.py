import dlib
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from skimage import io
from constants import *

# Convert landmarks to np
def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Resize image
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Set landmarks on image
def set_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    dets = detector(image, 1)
    if len(dets) == 0:
        dets = detector(image, 2)
        if len(dets) == 0:
            dets = detector(image, 3)
            if len(dets) == 0:
                return None
    print('[+] Number of faces: {}'.format(len(dets)))

    for k, d in enumerate(dets):
        shape = predictor(image, d)
        shape = shape_to_np(shape)
        height, width = image.shape
        image = np.zeros((height, width, 1), np.uint8)        

        for(x, y) in shape:
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)

        image = resize(image, width=SIZE_FACE, height=SIZE_FACE)
        if len(dets) > 1:
            break;

    return image

# Prepare label
def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

# Prepare image
def data_to_image(data):
    data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = set_landmarks(data_image)
    return data_image

# Load csv
data = pd.read_csv(DATASET_PATH)

data_images = []
data_labels = []
test_images = []
test_labels = []
index = 1
total = data.shape[0]

# Load images and labels from csv into arrays and save file
for index, row in data.iterrows():
    emotion = emotion_to_vec(row['emotion'])
    image = data_to_image(row['pixels'])    
    if image is not None:
        if index <= TRAINING_SIZE:
            data_labels.append(emotion)
            data_images.append(image)
        else:
            test_labels.append(emotion)
            test_images.append(image)
    else:
        print("[+] Image is null. Error!")
    index += 1
    print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print("Training samples: " + str(len(data_images)))
print("Validation samples: " + str(len(test_images)))
print("Total: " + str(len(data_images) + len(test_images)))
np.save(SAVE_DATASET_IMAGES_LANDMARKS_FILENAME, data_images)
np.save(SAVE_DATASET_LABELS_LANDMARKS_FILENAME, data_labels)
np.save(SAVE_DATASET_IMAGES_TEST_LANDMARKS_FILENAME, test_images)
np.save(SAVE_DATASET_LABELS_TEST_LANDMARKS_FILENAME, test_labels)