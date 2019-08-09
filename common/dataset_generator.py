import html
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common.client import AnimeApiClient

random.seed(12345)


class DatasetGenerator:
    def __init__(self, api_client: AnimeApiClient):
        self.api_client = api_client
        self.anime_lists = {
            'training': [],
            'validation': [],
        }
        self.vectorizer = None
        self.inverse_vectorizer = None
        self.selector = None
        self.dataset = {
            'training': {
                'ids': [],
                'data': np.array([]),
                'labels': [],
            },
            'validation': {
                'ids': [],
                'data': np.array([]),
                'labels': [],
            }
        }
        self.metadata = {}

    def get_vectorized_dataset(self, set_name, max_df=0.4, min_df=4):
        self.dataset[set_name]['data'] = self._vectorize_synopses(
            set_name=set_name,
            synopses=[anime['sanitized_synopsis'] for anime in self.anime_lists[set_name]],
            max_df=max_df,
            min_df=min_df)
        return self.dataset[set_name]

    def load_dataset(self, begin, end, validation_split=0.2):
        imported_anime = self.api_client.get_anime_range(begin, end)
        random.shuffle(imported_anime)
        self.metadata['total_num_media_queried'] = len(imported_anime)
        pruned_imported_anime = [
            sanitized_anime
            for sanitized_anime
            in (
                self._sanitize_synopsis(anime)
                for anime in imported_anime
            )
            if sanitized_anime['sanitized_synopsis_length'] > 10
        ]
        total_synopsis_length = sum(anime['sanitized_synopsis_length'] for anime in pruned_imported_anime)
        self.anime_lists['training'], self.anime_lists['validation'] = self._train_val_split(data=pruned_imported_anime, validation_split=validation_split)
        for set_name in ['training', 'validation']:
            self.dataset[set_name]['ids'] = [anime['id'] for anime in self.anime_lists[set_name]]
            self.dataset[set_name]['labels'] = [self._is_lewd(anime) for anime in self.anime_lists[set_name]]
            self.metadata['num_media_in_{}_set'.format(set_name)] = len(self.dataset[set_name]['ids'])
            self.metadata['num_lewd_media_in_{}_set'.format(set_name)] = sum(self.dataset[set_name]['labels'])
        self.metadata['total_num_media_kept'] = len(pruned_imported_anime)
        self.metadata['total_num_media_discarded'] = self.metadata['total_num_media_queried'] - self.metadata['total_num_media_kept']
        self.metadata['average_synopsis_length'] = total_synopsis_length / self.metadata['total_num_media_kept']
        self.metadata['num_media_in_validation_set'] = len(self.dataset['validation']['ids'])
        return self.metadata

    def _vectorize_synopses(self, synopses, set_name, max_df, min_df):
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(**{
                'ngram_range': (1, 2),
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': 'word',
                'max_df': max_df,
                'min_df': min_df,
            })
        if set_name == 'training':
            self.vectorizer.fit(synopses)
        self.dataset[set_name]['data'] = self.vectorizer.transform(synopses).astype('float32')
        self.metadata['num_tokens'] = self.dataset[set_name]['data'].shape[1]
        self.metadata['{}_set_data_vector_shape'.format(set_name)] = self.dataset[set_name]['data'].shape
        return self.dataset[set_name]['data']

    @staticmethod
    def _train_val_split(data, validation_split):
        return data[int(round(len(data) * validation_split)):], data[:int(round(len(data) * validation_split))]

    def vector_to_ngram(self, index):
        if not self.inverse_vectorizer:
            self.inverse_vectorizer = {v: k for k, v in self.vectorizer.vocabulary_.items()}
        return self.inverse_vectorizer[index]

    @staticmethod
    def _is_lewd(anime):
        return bool(anime['is_nsfw'] or "Ecchi" in anime['tags'])

    @staticmethod
    def _sanitize_synopsis(anime):
        if not anime['synopsis']:
            anime['sanitized_synopsis'] = ''
            anime['sanitized_synopsis_length'] = 0
            return anime
        anime['sanitized_synopsis'] = anime['synopsis'].strip()
        anime['sanitized_synopsis'] = html.unescape(anime['sanitized_synopsis'])
        # Remove URLs
        anime['sanitized_synopsis'] = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', anime['sanitized_synopsis'])
        # Remove html elements
        anime['sanitized_synopsis'] = re.sub(r'<[\w\/="!\s]+?>', '', anime['sanitized_synopsis'])
        # If the line contains source and a colon, delete source and everything after, as well as parentheses if they exist
        anime['sanitized_synopsis'] = re.sub(r'[\[\(]?\s*Source?\s*:.{0,40}\s*$', '', anime['sanitized_synopsis'], flags=re.IGNORECASE | re.MULTILINE)
        # If the line contains source and a parentheses, delete it and everything after
        anime['sanitized_synopsis'] = re.sub(r'[\[\(]\s*Source?.{0,40}\s*$', '', anime['sanitized_synopsis'], flags=re.IGNORECASE | re.MULTILINE)
        # If the line contains from and a weird character in front of it, delete from and everything after
        anime['sanitized_synopsis'] = re.sub(r'[~\[\(]\s*from.{0,40}\s*$', '', anime['sanitized_synopsis'], flags=re.IGNORECASE | re.MULTILINE)
        anime['sanitized_synopsis'] = re.sub(r'\'’', '', anime['sanitized_synopsis'])
        anime['sanitized_synopsis'] = re.sub(r'[^a-zA-Z]', ' ', anime['sanitized_synopsis']).lower().strip()
        anime['sanitized_synopsis_length'] = len(anime['sanitized_synopsis'].split())
        return anime
