import numpy as np
import tensorflow as tf


def evaluate_model(model: tf.keras.Model, dataset, **kwargs):
    return model.evaluate(dataset['data'], dataset['labels'], **kwargs)


def get_mean_weights(model, layer=1):
    weights, biases = model.layers[layer].get_weights()
    print('Layer={}, weights.shape={}'.format(layer, weights.shape), flush=True)
    mean_weights = [sum(row) / len(row) for row in weights]
    return mean_weights


def get_sorted_token_weights(weights, index_to_token):
    sorted_indexes = np.argsort(weights)
    mean_token_weights = {index_to_token(i): weights[i] for i in sorted_indexes}
    return mean_token_weights
