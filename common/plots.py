import inflection
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import io


def plot_history_metric(history_dict, metric_name, save_dir):
    metric = history_dict[metric_name]
    val_metric = history_dict['val_{}'.format(metric_name)]
    epochs = range(1, len(metric) + 1)
    plt.plot(epochs, metric, '--g', label='Training {}'.format(inflection.humanize(metric_name)))
    plt.plot(epochs, val_metric, '--r', label='Validation {}'.format(inflection.humanize(metric_name)))
    # plt.plot(epochs, acc, '-g', label='Training acc')
    # plt.plot(epochs, val_acc, '-r', label='Validation acc')
    plt.title('{} over time'.format(metric_name))
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    metric_file_path = '{}{}.svg'.format(save_dir, inflection.underscore(metric_name))
    if save_dir:
        io.gfile.makedirs(save_dir)
        plt.savefig(metric_file_path)
    plt.legend()
    plt.show()
    return metric_file_path


def plot_weights(token_weights, save_dir, title="Token Weights", ):
    plot_range = np.arange(len(token_weights))
    plt.bar(plot_range, [v for k, v in token_weights], align='center', alpha=0.5)
    plt.xticks(plot_range, [k for k, v in token_weights], rotation=45, ha='right', fontsize='x-small')
    plt.ylabel('Weight')
    plt.title(title)
    file_name = inflection.underscore(title)
    weights_file_path = '{}{}.svg'.format(save_dir, file_name)
    if save_dir:
        io.gfile.makedirs(save_dir)
        plt.savefig(weights_file_path)
    plt.show()
    return weights_file_path
