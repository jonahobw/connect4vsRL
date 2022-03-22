# %matplotlib inline
import pathlib

from .config import *
import platform
import numpy as np

import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    BatchNormalization,
    LeakyReLU,
    add,
)

# from keras.optimizers import SGD
from keras.optimizers import gradient_descent_v2

SGD = gradient_descent_v2.SGD

from keras import regularizers
import tensorflow as tf

# import tflite_runtime.interpreter as tflite

from .loss import softmax_cross_entropy_with_logits
from .settings import run_folder, run_archive_folder


class Gen_Model:
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, tflite=False):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tflite = tflite

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        if self.tflite:
            raise ValueError("Cannot do backprop on tflite model.")
        return self.model.fit(
            states,
            targets,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            batch_size=batch_size,
        )

    def write(self, version):
        if self.tflite:
            raise ValueError("Write not supported on tflite model.")
        else:
            self.model.save(
                run_archive_folder
                + "models/version"
                + "{0:0>4}".format(version)
                + ".h5"
            )

    def read(self, game, run_number, version):
        prefix = (
            run_archive_folder
            + game
            + "/run"
            + str(run_number).zfill(4)
            + "/models/version"
            + "{0:0>4}".format(version)
        )
        if self.tflite:
            raise ValueError("Read not supported on tflite model.")
        return load_model(
            prefix + ".h5",
            custom_objects={
                "softmax_cross_entropy_with_logits": softmax_cross_entropy_with_logits
            },
        )

    def printWeightAverages(self):
        layers = self.model.layers
        for i, l in enumerate(layers):
            try:
                x = l.get_weights()[0]
                print(
                    "WEIGHT LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f",
                    i,
                    np.mean(np.abs(x)),
                    np.std(x),
                    np.max(np.abs(x)),
                    np.min(np.abs(x)),
                )
            except:
                pass
        print("------------------")
        for i, l in enumerate(layers):
            try:
                x = l.get_weights()[1]
                print(
                    "BIAS LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f",
                    i,
                    np.mean(np.abs(x)),
                    np.std(x),
                    np.max(np.abs(x)),
                    np.min(np.abs(x)),
                )
            except:
                pass
        print("******************")

    def viewLayers(self):
        layers = self.model.layers
        for i, l in enumerate(layers):
            x = l.get_weights()
            print("LAYER " + str(i))

            try:
                weights = x[0]
                s = weights.shape

                fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
                channel = 0
                filter = 0
                for i in range(s[2] * s[3]):
                    sub = fig.add_subplot(s[3], s[2], i + 1)
                    sub.imshow(
                        weights[:, :, channel, filter],
                        cmap="coolwarm",
                        clim=(-1, 1),
                        aspect="auto",
                    )
                    channel = (channel + 1) % s[2]
                    filter = (filter + 1) % s[3]

            except:

                try:
                    fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
                    for i in range(len(x)):
                        sub = fig.add_subplot(len(x), 1, i + 1)
                        if i == 0:
                            clim = (0, 2)
                        else:
                            clim = (0, 2)
                        sub.imshow([x[i]], cmap="coolwarm", clim=clim, aspect="auto")

                    plt.show()

                except:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(x[0], cmap="coolwarm", clim=(-1, 1), aspect="auto")

                        plt.show()

                    except:
                        pass

            plt.show()

        print("------------------")

    def convertToModelInput(self, state):
        inputToModel = (
            state.binary
        )  # np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
        inputToModel = np.reshape(inputToModel, self.input_dim)
        return inputToModel


class Residual_CNN(Gen_Model):
    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers):
        Gen_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model = self._build_model()

    def residual_layer(self, input_block, filters, kernel_size):

        x = self.conv_layer(input_block, filters, kernel_size)

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)

        x = BatchNormalization(axis=1)(x)

        x = add([input_block, x])

        x = LeakyReLU()(x)

        return x

    def conv_layer(self, x, filters, kernel_size):

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def value_head(self, x):

        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            20,
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)

        x = LeakyReLU()(x)

        x = Dense(
            1,
            use_bias=False,
            activation="tanh",
            kernel_regularizer=regularizers.l2(self.reg_const),
            name="value_head",
        )(x)

        return x

    def policy_head(self, x):

        x = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding="same",
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        x = Dense(
            self.output_dim,
            use_bias=False,
            activation="linear",
            kernel_regularizer=regularizers.l2(self.reg_const),
            name="policy_head",
        )(x)

        return x

    def _build_model(self):

        main_input = Input(shape=self.input_dim, name="main_input")

        x = self.conv_layer(
            main_input,
            self.hidden_layers[0]["filters"],
            self.hidden_layers[0]["kernel_size"],
        )

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h["filters"], h["kernel_size"])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(
            loss={
                "value_head": "mean_squared_error",
                "policy_head": softmax_cross_entropy_with_logits,
            },
            optimizer=SGD(lr=self.learning_rate, momentum=MOMENTUM),
            loss_weights={"value_head": 0.5, "policy_head": 0.5},
        )

        return model


class Residual_CNN_tflite(Gen_Model):
    EDGETPU_SHARED_LIB = {
        "Linux": "libedgetpu.so.1",
        "Darwin": "libedgetpu.1.dylib",
        "Windows": "edgetpu.dll",
    }[platform.system()]

    def __init__(self, reg_const, learning_rate, input_dim, output_dim, hidden_layers):
        Gen_Model.__init__(
            self, reg_const, learning_rate, input_dim, output_dim, tflite=True
        )
        self.hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers)
        self.model_path = None
        self.interpreter = None

    def read(self, game, run_number, version):
        name = "version" + "{0:0>4}".format(version) + ".tflite"

        path = pathlib.Path.cwd() /"tensorflow_model" / "run_archive" / game / f"run{str(run_number).zfill(4)}" / "models" / name

        if not path.exists():
            raise FileNotFoundError(f"model file {path} not found")

        self.model_path = str(path)

        if TPU:
            self.interpreter = tf.lite.Interpreter(
                self.model_path,
                experimental_delegates=[
                    tf.lite.experimental.load_delegate(self.EDGETPU_SHARED_LIB)
                ],
            )
        else:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
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
