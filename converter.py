import tensorflow as tf
import os

def _to_tflite(modelName, tfliteName):
    model = tf.keras.models.load_model('models\\{}.h5'.format(modelName))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('tflite\\{}.tflite'.format(tfliteName), "wb").write(tflite_model)

def _copy_labels():
    f = open('models\\labels.txt', 'r')
    lines = f.readlines()
    f.close()
    f = open('tflite\\labels.txt', 'w')
    f.writelines(lines)
    f.close()

def convert_tflite():
    if(not os.path.isdir('tflite')):
        os.mkdir('tflite')

    #_to_tflite('facenet_keras', 'facenet')
    _to_tflite('faces', 'model')
    _copy_labels()

