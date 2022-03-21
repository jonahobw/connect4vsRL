from model import Residual_CNN
import config
from game import Game

import tensorflow as tf

env = Game()

model = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                              config.HIDDEN_CNN_LAYERS)

model_network = model.read(env.name, run_number=0, version=32)
model.model.set_weights(model_network.get_weights())

converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
tflite_model = converter.convert()

with open('run_archive/connect4/run0000/models/version0032.tflite', 'wb') as f:
  f.write(tflite_model)