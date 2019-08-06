import warnings
from pprint import PrettyPrinter

import numpy as np

MAX_NUM_WORDS = 50000

warnings.simplefilter(action='ignore', category=FutureWarning)
from dataset import AnimeDatasetGenerator
from classifier import BinaryTextClassifierModel
from tensorflow import keras

pp = PrettyPrinter(compact=True)

kitsu_datatset_generator = AnimeDatasetGenerator()

print("Creating training dataset...")
kitsu_datatset_generator.load_dataset(begin=0, end=7000, num_words=MAX_NUM_WORDS)
training_dataset = kitsu_datatset_generator.get_processed_dataset()
training_anime_list = kitsu_datatset_generator.anime_list

print("Creating testing dataset...")
kitsu_datatset_generator.load_dataset(begin=7000, end=10000, num_words=MAX_NUM_WORDS)
testing_dataset = kitsu_datatset_generator.get_processed_dataset()
testing_anime_list = kitsu_datatset_generator.anime_list

btc_model = BinaryTextClassifierModel(num_words=MAX_NUM_WORDS).get_model()

divider = 5 * len(training_dataset['data']) // 6

history = btc_model.fit(training_dataset['data'][:divider],
                        training_dataset['is_isekai'][:divider],
                        epochs=60,
                        batch_size=512,
                        validation_data=(training_dataset['data'][divider:], training_dataset['is_isekai'][divider:]),
                        verbose=1)

results = btc_model.test(x=testing_dataset['data'], y=testing_dataset['is_isekai'], verbose=True)

print(results)

full_ids = np.concatenate((testing_dataset['ids'], training_dataset['ids']))
full_data = np.concatenate((testing_dataset['data'], training_dataset['data']))
full_anime_list = {**training_anime_list, **testing_anime_list}

predictions = btc_model.predict(testing_dataset['data'])


def _convert_integer_list_to_string(list):
    return " ".join(kitsu_datatset_generator.index_to_ngram[index] for index in list)


def get_top_n_predictions(predictions_list, n):
    sorted_indices = sorted(range(len(predictions_list)), key=lambda index: predictions_list[index])
    return sorted_indices[-n:]


top_predictions = get_top_n_predictions(predictions, 15)
for i in top_predictions:
    pp.pprint({
        'id': testing_dataset['ids'][i],
        'prediction': predictions[i],
        'anime': full_anime_list.get(testing_dataset['ids'][i])
    })
    print('-----------------------------------------------\n')
