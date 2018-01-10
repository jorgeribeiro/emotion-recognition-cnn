import cv2
from constants import *
from emotion_recognition_cnn_training import EmotionRecognition
import numpy as np

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def format_image(image):
	if len(image.shape) > 2 and image.shape[2] == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	faces = cascade_classifier.detectMultiScale(image, scaleFactor = 1.3, minNeighbors = 5)

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

network = EmotionRecognition()
network.build_network()

video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	ret, frame = video_capture.read()

	result = network.predict(format_image(frame))

	if result is not None:
		for index, emotion in enumerate(EMOTIONS):
			cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
			cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

	cv2.imshow('Emotion Recognition', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_capture.release()
cv2.destroyAllWindows()