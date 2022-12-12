from tensorflow.keras.models import load_model
from neural_network import generate_nn
from converter import Converter
from data import Data
import mtcnn

facenet = load_model('models\\facenet_keras.h5')
detector = mtcnn.MTCNN()
# Cria o dataset
Data(facenet, detector, 'dataset').generate_data()
# Usa o dataset known.csv criado acima e o unknown.csv que já existia
# para treinar a rede neural e criar o modelo
generate_nn('dataset\\unknown.csv', 'dataset\\known.csv')
# Converte os modelos criados em tflite para serem baixados pelo client
Converter('models', 'tflite').convert_tflite()