import tensorflow as tf
import os

class Converter:
    def __init__(self, modelsDir, tfliteDir):
        self.modelsDir = modelsDir
        self.tfliteDir = tfliteDir

    def _to_tflite(self, modelName, tfliteName):
        model = tf.keras.models.load_model('{}\\{}.h5'.format(self.modelsDir, modelName))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open('{}\\{}.tflite'.format(self.tfliteDir, tfliteName), "wb").write(tflite_model)

    def _copy_labels(self):
        f = open('{}\\labels.txt'.format(self.modelsDir), 'r')
        lines = f.readlines()
        f.close()
        f = open('{}\\labels.txt'.format(self.tfliteDir), 'w')
        f.writelines(lines)
        f.close()

    def convert_tflite(self):
        if(not os.path.isdir('tflite')):
            os.mkdir('tflite')

        # Como o modelo facenet já é treinado anteriormente, eu já mantenho
        # uma cópia dele nos clients, então não precisa gerar tflite dele novamente
        # apenas do modelo de reconhcimento e das labels que são mudadas a cada nova pessoa

        # _to_tflite('facenet_keras', 'facenet')
        self._to_tflite('faces', 'model')
        self._copy_labels()

