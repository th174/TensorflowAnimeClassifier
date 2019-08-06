import os
import numpy as np
from tensorboard import summary
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

MODEL_DIRECTORY = "./models/"


class BinaryTextClassifier:
    def __init__(self, num_features, layers=2, units=64, dropout_rate=0.2, learning_rate=1e-3):
        print('Hyperparameters:')
        print({
            'dense_layers': layers,
            'hidden_units': units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
        })
        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Dropout(rate=0, input_shape=(num_features,))
        )
        for _ in range(layers - 1):
            self.model.add(
                keras.layers.Dense(
                    units=units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.002)
                )
            )
            self.model.add(
                keras.layers.Dropout(rate=dropout_rate)
            )
        self.model.add(
            keras.layers.Dense(
                units=1,
                activation='sigmoid',
            )
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            metrics=['accuracy', 'Precision', 'Recall'],
            loss='binary_crossentropy',
        )
        self.model.summary()
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)]
        self.history_dict = None

    def train(self, dataset, validation_split=0.2, epochs=1000, batch_size=512, verbose=1, **kwargs):
        self.history_dict = self.model.fit(
            x=dataset['data'],
            y=dataset['labels'],
            validation_split=validation_split,
            callbacks=self.callbacks,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs).history
        print('Validation accuracy: {acc}, loss: {loss}'.format(acc=self.history_dict['val_acc'][-1], loss=self.history_dict['val_loss'][-1]))
        return self.history_dict['val_acc'][-1], self.history_dict['val_loss'][-1]

    def plot_history_loss(self):
        acc = self.history_dict['acc']
        val_acc = self.history_dict['val_acc']
        loss = self.history_dict['loss']
        val_loss = self.history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, loss, '--g', label='Training loss')
        plt.plot(epochs, val_loss, '--r', label='Validation loss')
        # plt.plot(epochs, acc, '-g', label='Training acc')
        # plt.plot(epochs, val_acc, '-r', label='Validation acc')
        plt.title('Binary Crossentropy over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_history_accuracy(self):
        acc = self.history_dict['acc']
        val_acc = self.history_dict['val_acc']
        # loss = self.history_dict['loss']
        # val_loss = self.history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        # plt.plot(epochs, loss, '--g', label='Training loss')
        # plt.plot(epochs, val_loss, '--r', label='Validation loss')
        plt.plot(epochs, acc, '-g', label='Training acc')
        plt.plot(epochs, val_acc, '-r', label='Validation acc')
        plt.title('Accuracy over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def get_token_index_weights(self, vector_to_ngram, layer=1):
        weights, biases = self.model.layers[layer].get_weights()
        print(f'Layer={layer}, weights.shape={weights.shape}')
        mean_weights = [sum(row) / len(row) for row in weights]
        sorted_indexes = np.argsort(mean_weights)
        mean_token_weights = {vector_to_ngram(i): mean_weights[i] for i in sorted_indexes}
        return mean_token_weights

    @staticmethod
    def plot_token_weights(token_weights, title):
        plot_range = np.arange(len(token_weights))
        plt.bar(plot_range, [v for k, v in token_weights], align='center', alpha=0.5)
        plt.xticks(plot_range, [k for k, v in token_weights], rotation=50, ha='right', fontsize='x-small')
        plt.ylabel('Weight')
        plt.title(title)
        plt.savefig('./plots/weights.svg')
        plt.show()

    def test(self, dataset, **kwargs):
        return self.model.evaluate(dataset['data'], dataset['labels'], **kwargs)

    def save(self, name, save_dir='./models'):
        self.model.save(f'{save_dir}/{name}.h5')
