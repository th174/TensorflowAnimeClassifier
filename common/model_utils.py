import numpy as np
import tensorflow as tf


def compile_model(model, metrics=('AUC', 'BinaryAccuracy', 'Precision', 'Recall'), loss='binary_crossentropy', learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        metrics=metrics,
        loss=loss,
    )
    model.summary()


def save_model(model: tf.keras.Model, export_path):
    return model.save(export_path)


def evaluate_model(model, dataset, **kwargs):
    return model.evaluate(dataset['data'], dataset['labels'], **kwargs)


def get_mean_weights(model, layer=1):
    weights, biases = model.layers[layer].get_weights()
    print('Layer={}, weights.shape={}'.format(layer, weights.shape))
    mean_weights = [sum(row) / len(row) for row in weights]
    return mean_weights


def get_sorted_token_weights(weights, index_to_token):
    sorted_indexes = np.argsort(weights)
    mean_token_weights = {index_to_token(i): weights[i] for i in sorted_indexes}
    return mean_token_weights
