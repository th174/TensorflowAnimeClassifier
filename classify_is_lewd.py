from pprint import PrettyPrinter
import tensorflow as tf
from anilist import Anilist
from dataset import AnimeDatasetGenerator
from classifier import BinaryTextClassifier

tf.logging.set_verbosity(tf.logging.INFO)

pp = PrettyPrinter(compact=True)

dataset_generator = AnimeDatasetGenerator(api_client=Anilist())

MAX_NGRAMS = 20000

print("Creating dataset")
metadata = dataset_generator.load_dataset(begin=0, end=None, max_ngrams=MAX_NGRAMS)
pp.pprint(metadata)

# testing_set_size = len(dataset_generator.anime_list) // 5
testing_set_size = 0

print("Preparing training dataset")
training_dataset = dataset_generator.get_vectorized_dataset(_slice=slice(testing_set_size, None))

print("Building model")
btc_model = BinaryTextClassifier(num_features=training_dataset['data'].shape[1],
                                 layers=2,
                                 units=256,
                                 dropout_rate=0.3,
                                 learning_rate=5e-4)

print("Proceeding to train model")
btc_model.train(
    dataset=training_dataset,
    validation_split=0.2,
    epochs=500,
    batch_size=128,
    verbose=2,
)

btc_model.save("lewd_anime_classifier")

# print("Preparing testing dataset")
# testing_dataset = dataset_generator.get_vectorized_dataset(_slice=slice(0, testing_set_size))
#
# print("Proceeding to evaluate model")
# btc_model.test(
#     dataset=testing_dataset
# )

print("Generating plots")
btc_model.plot_history_loss()

btc_model.plot_history_accuracy()

token_weights = btc_model.get_token_index_weights(
    vector_to_ngram=dataset_generator.vector_to_ngram,
    layer=1
)

sorted_tokens = sorted(token_weights.keys(), key=lambda key: token_weights[key])

btc_model.plot_token_weights(
    title='Most positive tokens (Layer 1)',
    token_weights=[(token, token_weights[token]) for token in sorted_tokens[-20:]],
)

btc_model.plot_token_weights(
    title='Most negative tokens (Layer 1)',
    token_weights=[(token, token_weights[token]) for token in sorted_tokens[:20]],
)


# token_weights = btc_model.get_token_index_weights(
#     vector_to_ngram=dataset_generator.vector_to_ngram,
#     layer=5
# )
#
# sorted_tokens = sorted(token_weights.keys(), key=lambda key: token_weights[key])
#
# btc_model.plot_token_weights(
#     title='Most positive tokens (Layer 5)',
#     token_weights=[(token, token_weights[token]) for token in sorted_tokens[-20:]],
# )
#
# btc_model.plot_token_weights(
#     title='Most negative tokens (Layer 5)',
#     token_weights=[(token, token_weights[token]) for token in sorted_tokens[:20]],
# )
#
#

