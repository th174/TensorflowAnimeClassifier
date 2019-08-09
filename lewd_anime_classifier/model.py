from tensorflow import keras


def build_model(input_dim, output_dim, hidden_units=(64,), dropout_rate=.3):
    keras.backend.set_learning_phase(False)
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(rate=dropout_rate, input_shape=(input_dim,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units=units,
                                     activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(units=output_dim, activation='sigmoid'))
    return model
