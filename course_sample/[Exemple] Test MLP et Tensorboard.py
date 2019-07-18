from tensorflow.python.keras.models import *
from tensorflow.python.keras.activations import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.utils import *
from tensorflow.python.keras.metrics import *
from tensorflow.python.keras.datasets import *
from tensorflow.python.keras.callbacks import *
import tensorflow as tf
import numpy as np

file_writer = tf.contrib.summary.create_file_writer("/logs")
file_writer.set_as_default()


class AccDiffMetric(Callback):  # inherited keras.callbacks.Callback
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        acc_diff = logs['val_acc'] - logs['acc']
        logs['acc_diff'] = acc_diff
        tf.summary.scalar('Accuracy_difference', tensor=acc_diff)
        return acc_diff


def create_mlp(input_dim: int,
               hidden_layers_count: int,
               neurons_count_per_hidden_layer: int,
               output_dim):
    model = Sequential()
    for i in range(hidden_layers_count):
        if i == 0:
            model.add(Dense(neurons_count_per_hidden_layer,
                            activation=tanh,
                            input_dim=input_dim))
        else:
            model.add(Dense(neurons_count_per_hidden_layer,
                            activation=tanh))
    model.add(Dense(output_dim, activation=sigmoid))
    model.compile(optimizer=sgd(),
                  loss=mse,
                  metrics=["accuracy"])

    return model


if __name__ == "__main__":
    (x_train, y_train), (_, _) = mnist.load_data()

    small_x = x_train[:1000]
    small_y = y_train[:1000]

    print(small_x.shape)
    small_x = np.reshape(small_x, (1000, 28 * 28)) / 255.0
    small_y = to_categorical(small_y, 10)

    print(small_x[0:4])
    print(small_y[0:4])

    print(small_x.shape)
    print(small_y.shape)

    acc_diff_callback = AccDiffMetric()

    for l in range(4, 9):
        for power_n in range(9, 13):
            if power_n > 3 and l == 0:
                break
            model = create_mlp(784, l, 2**power_n, 10)

            model_name = "mlp_" + str(l) + "_" + str(2**power_n)

            tb_callback = TensorBoard("./logs/" + model_name)

            model.fit(small_x, small_y,
                      epochs=100,
                      batch_size=128,
                      validation_split=0.2,
                      callbacks=[acc_diff_callback, tb_callback])

            plot_model(model, "./models/" + model_name + ".png", show_shapes=True,
                       show_layer_names=True)
            model.save("./models/" + model_name)
