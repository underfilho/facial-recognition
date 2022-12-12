from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.utils import shuffle
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# Cria a arquitetura da rede neural e faz o treinamento dela
# Recebendo csv de face-id's conhecidos e desconhecidos
def generate_nn(unknown_csv_path, known_csv_path):
    # Carrega os dados
    unknown = pd.read_csv(unknown_csv_path)
    known = pd.read_csv(known_csv_path)

    # Concatena os dados conhecidos e desconhecidos
    df = pd.concat([unknown, known])
    # Em x irá conter apenas os vetores associados a cada pessoa
    X = df.drop('target', axis=1).to_numpy()
    # Em y irá conter os nomes das pessoas, que é o output esperado
    y = df.target.to_numpy()
    num_person = len(np.unique(y))

    # Então o objetivo da rede neural é basicamente através dos dados de exemplo no csv 
    # identificar como seria a melhor função que replique o funcionamento
    # e através da nova função fazer novas predições no client

    norm = Normalizer(norm='l2')
    X = norm.transform(X)
    
    # Depois de normalizar os dados faz um embaralhamento dos dados
    # pois geralmente vão estar ordenados por pessoa
    trainX, trainY = shuffle(X, y, random_state=40)
    out_encoder = LabelEncoder()
    out_encoder.fit(trainY)
    # Encoda os outputs, para que ao invés de nomes, sejam numeros
    trainY = out_encoder.transform(trainY)
    trainY = to_categorical(trainY)

    # Cria a arquitetura da rede neural, contendo um layer, um dropout e outro layer
    # o último layer contem um valor associado a cada pessoa a ser reconhecida
    # assim ele retorna uma porcentagem de ser cada um 
    # (como ele foi treinado 1-1 vai sempre tentar priorizar uma pessoa)
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(128,)),
        layers.Dropout(0.5),
        layers.Dense(num_person, activation='softmax')
    ])

    # Compila a rede com os otimizadores e função loss necessária
    # e passa os datasets de X e Y, alem do numero de épocas
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=20, batch_size=4)

    # Salva o modelo
    model.save('models\\faces.h5')

    # Pega os nomes das pessoas do encoder associadas com cada número de output
    # e salva em labels.txt
    f = open('models\\labels.txt', 'w')
    content = map(lambda e: '{}\n'.format(e), out_encoder.classes_)
    f.writelines(content)
    f.close()