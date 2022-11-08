from tensorflow.keras.models import load_model
from neural_network import generate_nn
from converter import convert_tflite
from data import Data
import mtcnn

facenet = load_model('models\\facenet_keras.h5')
detector = mtcnn.MTCNN()
Data(facenet, detector).generate_data()
generate_nn()
convert_tflite()
