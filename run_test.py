from tensorflow.keras.models import load_model
from os import walk
import numpy as np
import mtcnn
import cv2

def _readentry():
    path = 'testset'
    images = []

    for (_, _, filenames) in walk(path):
        for i in filenames:
            images.append(i)

    return images


def _generate_testset(array):
    embeddings = []
    for i in array:
        img = _preprocess('testset\\{}'.format(i))

        faceid = _get_embedding(img)
        embeddings.append(faceid)

    return np.asarray(embeddings)

def _preprocess(detector, imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    faces = detector.detect_faces(img)
    print(imgPath)
    x1, y1, w, h = faces[0]['box']
    x2, y2 = x1 + w, y1 + h
    newimg = img[y1:y2, x1:x2]
    newimg = cv2.resize(newimg, (160, 160), interpolation = cv2.INTER_AREA)
    
    return newimg

def _generate_embeddings(facenet, detector, array):
    embeddings = []
    for i in array:
        img = _preprocess('testset\\{}'.format(i))

        faceid = _get_embedding(img)
        embeddings.append(faceid)

    return np.asarray(embeddings)

def _get_embedding(facenet, face_pixels):
    face_pixels = face_pixels.astype('float32')
    face_pixels = (face_pixels - 128) / 128

    samples = np.expand_dims(face_pixels, axis=0)

    yhat = facenet.predict(samples)
    embedding = yhat[0]

    return embedding

def run(facenet, detector):
    faces = load_model('models\\faces.h5')
    images = _readentry()
    X = _generate_embeddings(facenet, detector, images)

    prediction = faces.predict(X)
    prediction = np.argmax(prediction, axis=1)

    f = open('models\\labels.txt', 'r')
    lines = f.readlines()
    prediction = map(lambda e: lines[e], prediction)
    for i in prediction:
        print(i)