import tensorflow as tf
from tensorflow import keras


def build_model(input_dim,
                output_dim,
                metrics=('AUC', 'BinaryAccuracy', 'Precision', 'Recall'),
                loss='binary_crossentropy',
                learning_rate=1e-3,
                hidden_units=(64,),
                dropout_rate=.3):
    keras.backend.set_learning_phase(False)
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(rate=dropout_rate, input_shape=(input_dim,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units=units,
                                     activation=tf.nn.relu,
                                     kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(units=output_dim, activation=tf.nn.sigmoid))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        metrics=metrics,
        loss=loss,
    )
    model.summary()
    return model
