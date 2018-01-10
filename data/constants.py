#                               __                    __             
#                              /\ \__                /\ \__          
#   ___    ___     ___     ____\ \ ,_\    __      ___\ \ ,_\   ____  
#  /'___\ / __`\ /' _ `\  /',__\\ \ \/  /'__`\  /' _ `\ \ \/  /',__\ 
# /\ \__//\ \L\ \/\ \/\ \/\__, `\\ \ \_/\ \L\.\_/\ \/\ \ \ \_/\__, `\
# \ \____\ \____/\ \_\ \_\/\____/ \ \__\ \__/.\_\ \_\ \_\ \__\/\____/
#  \/____/\/___/  \/_/\/_/\/___/   \/__/\/__/\/_/\/_/\/_/\/__/\/___/  .py
#
#

CASC_PATH = '../haarcascade_files/haarcascade_frontalface_default.xml'
DATASET_PATH = 'fer2013.csv'
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
SIZE_FACE = 48
TRAINING_SIZE = 28709
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
SAVE_DIRECTORY = './data/'
SAVE_MODEL_FILENAME = 'model_50_epochs_base_dataset_data_aug'
SAVE_MODEL_LANDMARKS_FILENAME = 'model_50_epochs_landmarks_dataset'
RUN_NAME = 'emotion_rec_run_50_epochs_base_dataset_data_aug'

SAVE_DATASET_IMAGES_FILENAME = 'data_set_fer2013.npy'
SAVE_DATASET_LABELS_FILENAME = 'data_labels_fer2013.npy'
SAVE_DATASET_IMAGES_TEST_FILENAME = 'test_set_fer2013.npy'
SAVE_DATASET_LABELS_TEST_FILENAME = 'test_labels_fer2013.npy'

SAVE_DATASET_IMAGES_LANDMARKS_CIRCLES_FILENAME = 'data_set_landmarks_circles_fer2013.npy'
SAVE_DATASET_LABELS_LANDMARKS_CIRCLES_FILENAME = 'data_labels_landmarks_circles_fer2013.npy'
SAVE_DATASET_IMAGES_TEST_LANDMARKS_CIRCLES_FILENAME = 'test_set_landmarks_circles_fer2013.npy'
SAVE_DATASET_LABELS_TEST_LANDMARKS_CIRCLES_FILENAME = 'test_labels_landmarks_circles_fer2013.npy'
 
SAVE_DATASET_IMAGES_LANDMARKS_LINES_FILENAME = 'data_set_landmarks_lines_fer2013.npy'
SAVE_DATASET_LABELS_LANDMARKS_LINES_FILENAME = 'data_labels_landmarks_lines_fer2013.npy'
SAVE_DATASET_IMAGES_TEST_LANDMARKS_LINES_FILENAME = 'test_set_landmarks_lines_fer2013.npy'
SAVE_DATASET_LABELS_TEST_LANDMARKS_LINES_FILENAME = 'test_labels_landmarks_lines_fer2013.npy'