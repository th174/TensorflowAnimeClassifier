import argparse
import os
from datetime import datetime
from pprint import PrettyPrinter

import inflection as inflection
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.python.lib.io import file_io

from common import model_utils, plots
from common.anilist import Anilist
from common.dataset_generator import DatasetGenerator
from lewd_anime_classifier import model as lewd_anime_classifier_model

tf.compat.v1.disable_eager_execution()

CACHE_PATH = '.cache/anilist'
METRICS = ['BinaryCrossentropy', 'AUC', 'BinaryAccuracy', 'Precision', 'Recall']
HIDDEN_UNITS = 16
NUM_LAYERS = 2
VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-5
MAX_EPOCHS = 1000
BATCH_SIZE = 64
POSITIVE_FEATURE_WEIGHT = 4
EARLY_STOPPING_PATIENCE = 25
MAX_DF = .4
MIN_DF = 20
CHECKPOINT_FILE_PATH = 'lac_checkpoint.{epoch:02d}.hdf5'
NAME = "lewd_anime_classifier_model"
BENCHMARK_TOKENS = (
    'sex',
    'porn',
    'love',
    'penis',
    'oppai',
    'breasts',
    'sister',
    'teacher',
    'brother',
    'sexuality',
    'maid',
    'hot',
    'loli',
    'rape',
    'hentai',
    'ecchi',
    'onii',
    'imouto',
    'girl',
    'boy',
    'women',
    'yuri',
    'yaoi',
    'story',
    'stories',
    'lust',
    'erotic',
    'harem'
)

pp = PrettyPrinter(compact=True)

TESTING_SET_SIZE = 0


def copy_file_to_gcs(job_dir, file_path):
    if not job_dir.startswith('gs://'):
        return
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def create_dataset(api_client, begin=TESTING_SET_SIZE, end=None, benchmark_tokens=BENCHMARK_TOKENS):
    print("Creating dataset")
    dataset_generator = DatasetGenerator(api_client=api_client)
    metadata = dataset_generator.load_dataset(begin=begin, end=end, validation_split=VALIDATION_SPLIT)
    print("Printing select freqs: ")
    print({
        token: sum(
            sum(anime['sanitized_synopsis'].count(token) for anime in dataset_generator.anime_lists[set_name])
            for set_name in ['training', 'validation']
        )
        for token in benchmark_tokens
    })
    print('Dataset Metadata: ')
    print(metadata)
    return dataset_generator


def train_and_evaluate(args):
    os.makedirs('{}/logs/train'.format(args.job_dir), exist_ok=True)

    checkpoint_path = CHECKPOINT_FILE_PATH
    cache_path = CACHE_PATH
    if not args.job_dir.startswith('gs://'):
        checkpoint_path = os.path.join(args.job_dir, checkpoint_path)
        cache_path = os.path.join(args.job_dir, cache_path)

    dataset = create_dataset(Anilist(cache_path))
    training_dataset = dataset.get_vectorized_dataset(set_name='training', max_df=args.max_df, min_df=args.min_df)
    validation_dataset = dataset.get_vectorized_dataset(set_name='validation', max_df=args.max_df, min_df=args.min_df)

    early_stop = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=EARLY_STOPPING_PATIENCE, verbose=1)
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')
    tb_log = TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    model = lewd_anime_classifier_model.build_model(
        input_dim=training_dataset['data'].shape[1],
        output_dim=1,
        hidden_units=[args.hidden_units] * args.num_layers,
        dropout_rate=args.dropout_rate
    )

    model_utils.compile_model(model, metrics=METRICS, learning_rate=args.learning_rate)

    history_dict = model.fit(
        x=training_dataset['data'],
        y=training_dataset['labels'],
        validation_data=(validation_dataset['data'], validation_dataset['labels']),
        validation_split=args.validation_split,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        class_weight={
            0: 1,
            1: args.positive_feature_weight,
        },
        callbacks=[checkpoint, tb_log, early_stop]
    ).history

    weights = model_utils.get_mean_weights(model, layer=1)
    token_weights = model_utils.get_sorted_token_weights(weights=weights, index_to_token=dataset.vector_to_ngram)

    local_job_dir = '.' if args.job_dir.startswith('gs://') else args.job_dir
    model.save(os.path.join(local_job_dir, '{}.hdf5'.format(args.name)))
    copy_file_to_gcs(
        args.job_dir,
        os.path.join(local_job_dir, '{}.hdf5'.format(args.name))
    )

    for metric in args.metrics:
        plots.plot_history_metric(history_dict=history_dict, metric_name=inflection.underscore(metric), save_dir=args.job_dir)

    sorted_tokens = sorted(token_weights.keys(), key=lambda key: token_weights[key])
    plots.plot_weights(
        token_weights=[(token, token_weights[token]) for token in sorted_tokens[-30:]],
        title="Most positive tokens",
        save_dir=args.job_dir)
    plots.plot_weights(
        token_weights=[(token, token_weights[token]) for token in sorted_tokens[:30]],
        title="Most negative tokens",
        save_dir=args.job_dir)

    model_utils.save_model(model, os.path.join(args.job_dir, 'export'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='local_output')
    parser.add_argument(
        '--name',
        type=str,
        default=NAME, )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate for SGD')
    parser.add_argument(
        '--hidden-units',
        type=int,
        default=HIDDEN_UNITS,
        help='Number of nodes in first layer of DNN')
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=MAX_EPOCHS,
        help='Maximum number of epochs on which to train')
    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=10,
        help='Checkpoint per n training epochs')
    parser.add_argument(
        '--min-df',
        type=int,
        default=MIN_DF)
    parser.add_argument(
        '--max_df',
        type=float,
        default=MAX_DF),
    parser.add_argument(
        '--dropout-rate',
        type=int,
        default=DROPOUT_RATE, )
    parser.add_argument(
        '--positive-feature-weight',
        type=float,
        default=POSITIVE_FEATURE_WEIGHT, )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=VALIDATION_SPLIT
    ),
    parser.add_argument(
        '--verbose',
        type=int,
        default=2
    )
    parser.add_argument(
        '--metrics',
        type=list,
        default=METRICS
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=NUM_LAYERS
    )

    train_and_evaluate(parser.parse_known_args()[0])
