import argparse
import os
from pprint import PrettyPrinter

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.python.lib.io import file_io

from anime_classifier.common import model_utils, plots
from anime_classifier.common.anilist import Anilist
from anime_classifier.common.dataset_generator import DatasetGenerator

tf.compat.v1.disable_eager_execution()

HIDDEN_UNITS = (8, 4)
NUM_LAYERS = 2
VALIDATION_SPLIT = 0.2
MAX_NGRAMS = 20000
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-4
MAX_EPOCHS = 80
BATCH_SIZE = 64
POSITIVE_FEATURE_WEIGHT = 4
EARLY_STOPPING_PATIENCE = 40
MAX_DF = .5
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
anilist = Anilist()

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
    os.makedirs(f'{args.job_dir}/logs/train', exist_ok=True)

    dataset = create_dataset(anilist)
    training_dataset = dataset.get_vectorized_dataset(set_name='training', max_ngrams=args.max_ngrams, max_df=args.max_df, min_df=args.min_df)
    validation_dataset = dataset.get_vectorized_dataset(set_name='validation', max_ngrams=args.max_ngrams, max_df=args.max_df, min_df=args.min_df)

    checkpoint_path = CHECKPOINT_FILE_PATH
    if not args.job_dir.startswith('gs://'):
        checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

    early_stop = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=EARLY_STOPPING_PATIENCE)
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')
    tb_log = TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    model = model_utils.model_fn(
        input_dim=training_dataset['data'].shape[1],
        output_dim=1,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate
    )

    model_utils.compile_model(model, learning_rate=args.learning_rate)

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
    model.save(os.path.join(local_job_dir, f'{args.name}.hdf5'))
    copy_file_to_gcs(
        args.job_dir,
        os.path.join(local_job_dir, f'{args.name}.hdf5')
    )

    plots.plot_history_accuracy(history_dict=history_dict, save_dir=args.job_dir)
    plots.plot_history_loss(history_dict=history_dict, save_dir=args.job_dir)
    sorted_tokens = sorted(token_weights.keys(), key=lambda key: token_weights[key])
    plots.plot_weights(
        token_weights=[(token, token_weights[token]) for token in sorted_tokens[-30:]],
        title="Most positive tokens",
        save_dir=args.job_dir)
    plots.plot_weights(
        token_weights=[(token, token_weights[token]) for token in sorted_tokens[:30]],
        title="Most negative tokens",
        save_dir=args.job_dir)

    model_utils.save_model(model, os.path.join(args.job_dir, f'export'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default=f'local_output')
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
        type=list,
        default=HIDDEN_UNITS,
        help='Number of nodes in the each layer of DNN')
    parser.add_argument(
        '--num-layers',
        type=int,
        default=NUM_LAYERS,
        help='Number of layers in DNN')
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
        '--max-ngrams',
        type=int,
        default=MAX_NGRAMS,
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

    train_and_evaluate(parser.parse_known_args()[0])
