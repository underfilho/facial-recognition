from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.utils import shuffle
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
import numpy as np

def generate_nn():
    unknown = pd.read_csv('dataset\\unknown.csv')
    known = pd.read_csv('dataset\\known.csv')

    df = pd.concat([unknown, known])
    X = df.drop('target', axis=1).to_numpy()
    y = df.target.to_numpy()
    num_person = len(np.unique(y))

    norm = Normalizer(norm='l2')
    X = norm.transform(X)

    trainX, trainY = shuffle(X, y, random_state=40)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainY)
    trainY = out_encoder.transform(trainY)

    trainY = to_categorical(trainY)

    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(128,)),
        layers.Dropout(0.5),
        layers.Dense(num_person, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=20, batch_size=4)

    model.save('models\\faces.h5')

    f = open('models\\labels.txt', 'w')
    content = map(lambda e: '{}\n'.format(e), out_encoder.classes_)
    f.writelines(content)
    f.close()