import pathlib
import time
import argparse

from .config import *
import platform
import numpy as np

if platform.system() == "Linux":
    import tflite_runtime.interpreter as tflite


class Residual_CNN_tflite:
    EDGETPU_SHARED_LIB = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    def __init__(self):
        self.input_dim = (2,6,7)
        self.model_path = None
        self.interpreter = None

    def convertToModelInput(self, state):
        return np.reshape(state, self.input_dim)

    def read(self, game, run_number, version):
        name = "version" + "{0:0>4}".format(version) + ".tflite"

        path = pathlib.Path.cwd() /"tensorflow_model" / "run_archive" / game / f"run{str(run_number).zfill(4)}" / "models" / name

        if not path.exists():
            raise FileNotFoundError(f"model file {path} not found")

        self.model_path = str(path)

        if TPU:
            self.interpreter = tflite.Interpreter(
                self.model_path,
                experimental_delegates=[
                    tflite.load_delegate(self.EDGETPU_SHARED_LIB)
                ],
            )
        else:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

    def predict(self, x):
        input = self.interpreter.get_input_details()
        output = self.interpreter.get_output_details()

        x = x.astype(np.float32)

        self.interpreter.set_tensor(input[0]["index"], x)
        self.interpreter.invoke()
        out1 = self.interpreter.get_tensor(output[0]["index"])
        out2 = self.interpreter.get_tensor(output[1]["index"])
        return [out2, out1]


def convert_state_binary(state, playerTurn=1):
    currentplayer_position = np.zeros(42, dtype=np.int)
    currentplayer_position[state == playerTurn] = 1

    other_position = np.zeros(42, dtype=np.int)
    other_position[state == -playerTurn] = 1

    position = np.append(currentplayer_position, other_position)

    return (position)

parser = argparse.ArgumentParser()

parser.add_argument("-n", type=int, default=100, help="Number of inferences to run.")
parser.add_argument("-cpu", action='store_true', help="run on cpu")

args = parser.parse_args()

if not args.cpu:
    TPU = True
else:
    TPU = False

model = Residual_CNN_tflite()
model.read("connect4", 0, 32)

state = convert_state_binary(np.zeros(42, type=int))

inputToModel = np.array(args.n * [model.convertToModelInput(state)])

start = time.time()
preds = model.predict(inputToModel)
elapsed_ms = (time.time() - start)/1000

print(f"Took {elapsed_ms} ms to run {args.n} inferences, avg {elapsed_ms/args.n}ms.")