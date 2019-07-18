import numpy as np
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
import pandas as pd

file_writer = tf.contrib.summary.create_file_writer("/logs")
file_writer.set_as_default()


class AccDiffMetric(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        acc_diff = logs['val_acc'] - logs['acc']
        logs['acc_diff'] = acc_diff
        tf.summary.scalar('Accuracy_difference', tensor=acc_diff)
        return acc_diff


def create_mlp_model(input_dim: int,
                     hidden_layers_counter: int,
                     neurons_count_per_hidden_layer: int,
                     output_dim):
    mlp_model = Sequential()
    for i in range(hidden_layers_counter):
        if i == 0:
            mlp_model.add(Dense(neurons_count_per_hidden_layer,
                                activation=tanh,
                                input_dim=input_dim))
        else:
            mlp_model.add(Dense(neurons_count_per_hidden_layer,
                                activation=tanh))
    mlp_model.add(Dense(output_dim, activation=sigmoid))
    mlp_model.compile(optimizer=sgd(),
                      loss=mse,
                      metrics=["accuracy"])

    return mlp_model


def prepare_set(entries_number : int, data):
    x = data[:, 1:]
    y = data[:, 0]

    x = np.reshape(x, (entries_number, 28 * 28)) / 255.
    y = to_categorical(y, 10)

    return x, y


if __name__ == "__main__":
    train_quantity = 15000
    test_quantity = 2000

    data_train = pd.read_csv('fashion_mnist/fashion-mnist_train.csv').to_numpy()[0:train_quantity]
    data_test = pd.read_csv('fashion_mnist/fashion-mnist_test.csv').to_numpy()[0:test_quantity]

    # x_train = data_train[:, 1:]
    # y_train = data_train[:, 0]
    #
    # x_train = np.reshape(x_train, (60000, 28 * 28)) / 255.
    # y_train = to_categorical(y_train, 10)

    (x_train, y_train) = prepare_set(train_quantity, data_train)
    (x_test, y_test) = prepare_set(test_quantity, data_test)

    # print(x_train[0:4])
    # print(y_train[0:4])

    print(x_train.shape)
    print(y_train.shape)

    acc_diff_callback = AccDiffMetric()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=5)

    for l in range(5, 9):
        for neuron_power in range(9, 12):
            # if neuron_power > 3 and l == 0:
            #     break
            model = create_mlp_model(784, l, 2 ** neuron_power, 10)

            model_name = "mlp_" + str(l) + "_" + str(2 ** neuron_power)

            tb_callback = TensorBoard("./logs/mlp/" + model_name)

            model.fit(x_train, y_train,
                      epochs=10000,
                      batch_size=512,
                      validation_data=(x_test, y_test),
                      callbacks=[acc_diff_callback, tb_callback, early_stopping])

            plot_model(model, "./models/mlp/" + model_name + ".png", show_shapes=True,
                       show_layer_names=True)
            model.save("./models/mlp/" + model_name)
