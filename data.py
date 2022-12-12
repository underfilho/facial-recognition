from tensorflow.keras.models import load_model
import pandas as pd
from os import walk
import numpy as np
import mtcnn
import cv2

# A partir das imagens vai gerar o dataset para a rede neural
# esse dataset será um csv que irá conter em cada linha
# um vetor de 128 posições (o "face-id") e o nome da pessoa associada na frente
class Data():
    def __init__(self, facenet, detector, csv_dir):
        self.facenet = facenet
        self.detector = detector
        self.csv_dir = csv_dir

    # Essa função detecta as faces na imagem, recorta a face
    # e redimensiona para 160x160 que é o input do facenet
    def _preprocess(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        faces = self.detector.detect_faces(img)
        x1, y1, w, h = faces[0]['box']
        x2, y2 = x1 + w, y1 + h
        newimg = img[y1:y2, x1:x2]
        newimg = cv2.resize(newimg, (160, 160), interpolation = cv2.INTER_AREA)
        
        return newimg

    # A partir da imagem preprocessada, normaliza os pixels
    # e roda no modelo facenet, gerando o "face-id", ou face embedding
    def _get_embedding(self, face_pixels):
        face_pixels = face_pixels.astype('float32')
        face_pixels = (face_pixels - 128) / 128

        samples = np.expand_dims(face_pixels, axis=0)

        yhat = self.facenet.predict(samples)
        embedding = yhat[0]

        return embedding

    # Recebe um dicionário como input {"nome_da_pessoa": [path_imagem]}
    # a partir disso preprocessa e gera o face-id de cada imagem
    # retorna uma lista de todos os face-id's e os nomes associados a eles
    def _generate_datasets(self, dict):
        embeddings = []
        names = []
        for key, images in dict.items():
            for path in images:
                img = self._preprocess(path)

                faceid = self._get_embedding(img)
                embeddings.append(faceid)
                names.append(key)

        return (np.asarray(embeddings), names)

    # O método principal, primeiro ele gera o dicionário usando a função _readentry
    # depois lê a lista dos face-id's e dos nomes associados como X e y
    # e então cria o dataframe e salva no csv
    def generate_data(self):
        dict = self._readentry()
        X, y = self._generate_datasets(dict)

        df = pd.DataFrame(data=X)
        df['target'] = y
        df.to_csv('{}\\known.csv'.format(self.csv_dir), index=False)

    # Essa função se encarrega de através das imagens em /faces
    # gerar o input para _generate_datasets(), é a mais provável de ser alterada
    # já que o modo de pegar as imagens vai ser diferente
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
                    entry[folder].append('{}\\{}\\{}'.format(path, folder, i))
        
        return entry