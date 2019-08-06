import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import nltk
from nltk.corpus import stopwords
from api_client import AnimeApiClient


class AnimeDatasetGenerator:
    def __init__(self, api_client: AnimeApiClient):
        self.api_client = api_client
        self.anime_list = []
        nltk.download('stopwords')
        self.vectorizer = TfidfVectorizer(**{
            'ngram_range': (1, 2),
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',
            'stop_words': stopwords.words('english'),
            'min_df': 5,
        })
        self.inverse_vectorizer = None
        self.dataset = {
            'ids': [],
            'data': np.array([]),
            'labels': [],
        }
        self.metadata = {
            'total_num_media_queried': 0,
            'total_num_media_discarded': 0,
            'total_num_media_kept': 0,
            'average_synopsis_length': 0,
            'data_vector_shape': (0, 0),
            'total_num_lewd_media': 0,
            'total_num_tokens': 0,
            'used_num_tokens': 0,
        }

    def get_vectorized_dataset(self, _slice):
        return {k: v[_slice] for k, v in self.dataset.items()}

    def load_dataset(self, begin, end, max_ngrams=30000):
        self.anime_list = self.api_client.get_anime_range(begin, end)
        random.shuffle(self.anime_list)
        self.metadata['total_num_media_queried'] = len(self.anime_list)
        synopses = []
        total_synopsis_length = 0
        for anime in self.anime_list[:]:
            sanitized_synopsis = self._sanitize_synopsis(anime['synopsis'])
            synopsis_length = len(sanitized_synopsis.split())
            if synopsis_length < 10:
                self.anime_list.remove(anime)
            else:
                total_synopsis_length += synopsis_length
                synopses.append(sanitized_synopsis)
        self.dataset['ids'] = [anime['id'] for anime in self.anime_list]
        self.dataset['labels'] = [self._is_lewd(anime) for anime in self.anime_list]
        self.dataset['data'] = self.vectorize_synopses(synopses=synopses, max_ngrams=max_ngrams)
        self.metadata['total_num_media_kept'] = len(self.anime_list)
        self.metadata['total_num_media_discarded'] = self.metadata['total_num_media_queried'] - self.metadata['total_num_media_kept']
        self.metadata['average_synopsis_length'] = total_synopsis_length / self.metadata['total_num_media_kept']
        self.metadata['data_vector_shape'] = self.dataset['data'].shape
        self.metadata['total_num_lewd_media'] = sum(self.dataset['labels'])
        self.metadata['used_num_tokens'] = self.dataset['data'].shape[1]
        return self.metadata

    def vectorize_synopses(self, synopses, max_ngrams):
        vectors = self.vectorizer.fit_transform(synopses)
        self.metadata['total_num_tokens'] = vectors.shape[1]
        selector = SelectKBest(f_classif, k=min(max_ngrams, vectors.shape[1]))
        selector.fit(vectors, self.dataset['labels'])
        return selector.transform(vectors).astype('float32')

    def vector_to_ngram(self, index):
        if not self.inverse_vectorizer:
            self.inverse_vectorizer = self.vectorizer.get_feature_names()
        return self.inverse_vectorizer[index]

    @staticmethod
    def _is_lewd(anime):
        return bool(anime['is_nsfw'] or "Ecchi" in anime['tags'])

    @staticmethod
    def _sanitize_synopsis(synopsis):
        if not synopsis:
            return ''
        regex_html_tags = r'<[/\w]{0,10}?>'
        synopsis = re.sub(regex_html_tags, '', synopsis)
        regex_author_citation = r'[\(\[].+?[\)\]]\s*$'
        synopsis = re.sub(regex_author_citation, '', synopsis)
        regex_non_alphabetic = r'[^a-zA-Z\']'
        synopsis = re.sub(regex_non_alphabetic, ' ', synopsis).lower().strip()
        return synopsis
