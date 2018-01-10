from constants import *
import cv2
import pandas as pd
import numpy as np
from PIL import Image

# Not using cascade method to find faces
# Maybe implement it later to improve performance
# UPDATE: implemented

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:,:] = 200
    gray_border[((150 / 2) - (SIZE_FACE/2)):((150/2)+(SIZE_FACE/2)), ((150/2)-(SIZE_FACE/2)):((150/2)+(SIZE_FACE/2))] = image
    # use line below if there is any error with line above and vice-versa
    # gray_border[int(((150 / 2) - (SIZE_FACE/2))):int(((150/2)+(SIZE_FACE/2))), int(((150/2)-(SIZE_FACE/2))):int(((150/2)+(SIZE_FACE/2)))] = image
    image = gray_border

    faces = cascade_classifier.detectMultiScale(image, scaleFactor = 1.02, minNeighbors = 2)

    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image

# Prepare label
def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

# Prepare image
def data_to_image(data):
    data_image = np.fromstring(str(data), dtype = np.uint8, sep = ' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    data_image = format_image(data_image)
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
np.save(SAVE_DATASET_IMAGES_FILENAME, data_images)
np.save(SAVE_DATASET_LABELS_FILENAME, data_labels)
np.save(SAVE_DATASET_IMAGES_TEST_FILENAME, test_images)
np.save(SAVE_DATASET_LABELS_TEST_FILENAME, test_labels)