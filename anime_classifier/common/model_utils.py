import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import builder as saved_model_builder, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def model_fn(input_dim, output_dim, hidden_units=(64,), dropout_rate=.3):
    keras.backend.set_learning_phase(False)
    model = keras.models.Sequential()
    model.add(keras.layers.Dropout(rate=dropout_rate, input_shape=(input_dim,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units=units,
                                     activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.005)))
        model.add(keras.layers.Dropout(rate=dropout_rate))
    model.add(keras.layers.Dense(units=output_dim, activation='sigmoid'))
    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        metrics=['accuracy', 'Precision', 'Recall'],
        loss='binary_crossentropy',
    )
    model.summary()


def save_model(model, export_path):
    builder = saved_model_builder.SavedModelBuilder(export_path)

    signature = predict_signature_def(inputs={'input': model.inputs[0]}, outputs={'is_lewd': model.outputs[0]})

    with tf.compat.v1.keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            })
        builder.save()


def train_model(model, dataset, epochs=1000, batch_size=512, verbose=1, early_stopping_patience=20, **kwargs):
    return model.fit(
        x=dataset['data'],
        y=dataset['labels'],
        # validation_data = (validation_data, validation_labels)
        # validation_split=validation_split,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1)],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        **kwargs).history


def evaluate_model(model, dataset, **kwargs):
    return model.model.evaluate(dataset['data'], dataset['labels'], **kwargs)


def get_mean_weights(model, layer=1):
    weights, biases = model.layers[layer].get_weights()
    print(f'Layer={layer}, weights.shape={weights.shape}')
    mean_weights = [sum(row) / len(row) for row in weights]
    return mean_weights


def get_sorted_token_weights(weights, index_to_token):
    sorted_indexes = np.argsort(weights)
    mean_token_weights = {index_to_token(i): weights[i] for i in sorted_indexes}
    return mean_token_weights
