from tensorflow.keras.models import load_model
import pandas as pd
from os import walk
import numpy as np
import mtcnn
import cv2

class Data():
    def __init__(self, facenet, detector):
        self.facenet = facenet
        self.detector = detector

    def _readentry(self):
        path = 'faces'
        entry = {}

        for (dirpath, dirnames, filenames) in walk(path):
            if dirpath == path:
                for folder in dirnames:
                    entry[folder] = []
            else:
                folder = dirpath[len(path) + 1:len(dirpath)]
                
                for i in filenames:
                    entry[folder].append(i)
        
        return entry

    def _preprocess(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        faces = self.detector.detect_faces(img)
        x1, y1, w, h = faces[0]['box']
        x2, y2 = x1 + w, y1 + h
        newimg = img[y1:y2, x1:x2]
        newimg = cv2.resize(newimg, (160, 160), interpolation = cv2.INTER_AREA)
        
        return newimg

    def _get_embedding(self, face_pixels):
        face_pixels = face_pixels.astype('float32')
        face_pixels = (face_pixels - 128) / 128

        samples = np.expand_dims(face_pixels, axis=0)

        yhat = self.facenet.predict(samples)
        embedding = yhat[0]

        return embedding

    def _generate_datasets(self, dict):
        embeddings = []
        names = []
        for key, value in dict.items():
            images = value
            for i in images:
                img = self._preprocess('faces\\{}\\{}'.format(key, i))

                faceid = self._get_embedding(img)
                embeddings.append(faceid)
                names.append(key)

        return (np.asarray(embeddings), names)

    def generate_data(self):
        dict = self._readentry()
        X, y = self._generate_datasets(dict)

        df = pd.DataFrame(data=X)
        df['target'] = y
        df.to_csv('dataset\\known.csv', index=False)