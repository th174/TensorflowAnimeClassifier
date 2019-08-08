import os

import matplotlib.pyplot as plt
import numpy as np


def plot_weights(token_weights, save_dir, title="Token Weights", ):
    plot_range = np.arange(len(token_weights))
    plt.bar(plot_range, [v for k, v in token_weights], align='center', alpha=0.5)
    plt.xticks(plot_range, [k for k, v in token_weights], rotation=45, ha='right', fontsize='x-small')
    plt.ylabel('Weight')
    plt.title(title)
    file_name = title.strip().replace(r'\s+', '_').lower()
    weights_file_path = f'{save_dir}/{file_name}.svg'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(weights_file_path)
    plt.show()
    return weights_file_path


def plot_history_loss(history_dict, save_dir):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, '--g', label='Training loss')
    plt.plot(epochs, val_loss, '--r', label='Validation loss')
    # plt.plot(epochs, acc, '-g', label='Training acc')
    # plt.plot(epochs, val_acc, '-r', label='Validation acc')
    plt.title('Binary Crossentropy over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    loss_file_path = f'{save_dir}/loss.svg'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(loss_file_path)
    plt.legend()
    plt.show()
    return loss_file_path


def plot_history_accuracy(history_dict, save_dir):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    # loss = history_dict['loss']
    # val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, loss, '--g', label='Training loss')
    # plt.plot(epochs, val_loss, '--r', label='Validation loss')
    plt.plot(epochs, acc, '-g', label='Training acc')
    plt.plot(epochs, val_acc, '-r', label='Validation acc')
    plt.title('Accuracy over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    accuracy_file_path = f'{save_dir}/accuracy.svg'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(accuracy_file_path)
    plt.legend()
    plt.show()
    return accuracy_file_path
