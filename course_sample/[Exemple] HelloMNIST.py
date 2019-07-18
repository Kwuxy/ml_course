from tensorflow.python import keras
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0

    print(y_train[0])
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(y_train[0])

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2048,
                                 activation=keras.activations.relu,
                                 input_dim=784))
    model.add(keras.layers.Dense(2048,
                                 activation=keras.activations.relu))
    model.add(keras.layers.Dense(1024,
                                 activation=keras.activations.relu))
    model.add(keras.layers.Dense(10,
                                 activation=keras.activations.sigmoid))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(),
                  metrics=[keras.metrics.categorical_accuracy])

    print(model.summary())

    model.fit(x_train, y_train, epochs=100, batch_size=1024,
              validation_data=(x_test, y_test))
