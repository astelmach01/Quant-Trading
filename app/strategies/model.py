import datetime
import os

import keras_tuner as kt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from sklearn import preprocessing
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


class FeedForwardNetwork:
    def __init__(self, data: pd.DataFrame, file_path: str):
        self.data = data
        self.model = tf.keras.models.load_model(file_path)

    # get the data over the past 2 days
    def prepare_data(self, interval_amt: int = 192):
        le = preprocessing.LabelEncoder()
        self.data['PVSRA'] = le.fit_transform(self.data['PVSRA'].tolist())

        self.data['Datetime'] = self.data.index.to_pydatetime()
        self.data['Time in Day'] = (self.data['Datetime'].astype(np.int64) // 10 ** 9) % 86400

        self.data['Target'] = self.data['Close'].shift(-1)
        self.data.dropna(inplace=True)

        X, y = [], []
        # 15 min intervals for 2 days
        start = len(df) - 1 - interval_amt
        for end in range(len(df) - 1, -1, -interval_amt):
            group = df.iloc[start: end]

            if len(group) != interval_amt:
                break

            y.append(group.pop('Target').to_numpy())
            _ = group.pop('Datetime')
            X.append(group.to_numpy())

            start -= interval_amt

        X = np.array(X)[::-1]
        y = np.array(y)[::-1]

        return X, y


"""## Make the Neural Network"""

if __name__ == "__main__":
    df = feather.read_feather('PVSRA_2015.feather')
    df.set_index('Date', drop=True, inplace=True)
    df['PVSRA'] = df['PVSRA'].astype(str)

    net = FeedForwardNetwork(df)

    X, y = net.prepare_data()



    normalizer = layers.Normalization(input_shape=X.shape[1:], axis=None)
    normalizer.adapt(X)

    model = build_and_compile_model(normalizer)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X,
        y,
        validation_split=0.25,
        epochs=1000,
        callbacks=[tensorboard_callback])


    def build_and_compile_model(hp):
        model = keras.Sequential()
        model.add(normalizer)

        hp_units = hp.Int('first layer', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))

        hp_units = hp.Int('second layer', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))

        hp_units = hp.Int('third layer', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))

        hp_units = hp.Int('fourth layer', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(keras.layers.Dense(10))

        model.add(layers.Dense(1))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      metrics=[keras.metrics.MeanAbsoluteError()])
        return model


    tuner = kt.Hyperband(build_and_compile_model,
                         objective="val_mean_absolute_error",
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X, y, epochs=200, validation_split=0.25, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model: tf.keras.Model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        X,
        y,
        validation_split=0.25,
        epochs=1000,
        callbacks=[tensorboard_callback])

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard

    model.save('tuned_model.h5')

    os.system('tensorboard --logdir .')
