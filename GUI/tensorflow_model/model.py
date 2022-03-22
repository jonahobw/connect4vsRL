import pathlib

from .config import *
import platform
import numpy as np

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
        inputToModel = (
            state.binary
        )  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
        inputToModel = np.reshape(inputToModel, self.input_dim)
        return inputToModel

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
