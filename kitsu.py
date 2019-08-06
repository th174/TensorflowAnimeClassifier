import errno
import os
import random
import sys

import requests
import pprint
import json

from api_client import AnimeApiClient

KITSU_API_AUTH_URL = "https://kitsu.io/api/oauth"
KITSU_API_BASE_URL = 'https://kitsu.io/api/edge'
ANIME_ENDPOINT = '/anime'
CATEGORIES_ENDPOINT = '/categories'

CACHE_LOCATION = './.cache'

KITSU_CLIENT_ID = 'dd031b32d2f56c990b1425efe6c42ad847e7fe3ab46bf1299f05ecd856bdb7dd'
KITSU_CLIENT_SECRET = '54d7307928f63414defd96399fc31ba847961ceaecef3a5fd93144e960c0e151'
KITSU_USERNAME = 'tianyue.huang174@gmail.com'
KITSU_PASSWORD = 'Shippaishita1'

pp = pprint.PrettyPrinter(compact=True)


class Kitsu(AnimeApiClient):
    def __init__(self,
                 username=KITSU_USERNAME,
                 password=KITSU_PASSWORD,
                 client_id=KITSU_CLIENT_ID,
                 client_secret=KITSU_CLIENT_SECRET):
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth = self._authenticate()

    def get_anime_range(self, begin, end, batch_size=1):
        anime_list = {}
        for i in range(begin, end, batch_size):
            if (i - begin) % 100 == 0:
                print(f'Currently querying\t\ti={i}\t\tbegin={begin}\t\tend={end}')
            try:
                batch = self._get_anime(i)
                anime_list += batch
            except IOError as e:
                print(e, file=sys.stderr)
                pass
        random.shuffle(anime_list)
        return anime_list

    def _authenticate(self):
        r = requests.post(f'{KITSU_API_AUTH_URL}/token',
                          params={
                              "grant_type": "password",
                              "username": self.username,
                              "password": self.password,
                              # "client_id": KITSU_CLIENT_ID,
                              # "client_secret": KITSU_CLIENT_SECRET
                          })
        return r.json()

    def _get_anime(self, anime_id):
        response_body = self._get_cache_or_fetch(f'{ANIME_ENDPOINT}/{anime_id}',
                                                 params={'include': 'categories'},
                                                 headers=self._get_auth_header())
        entry = response_body['data']
        return [
            {
                'id': entry['id'],
                'title': entry['attributes']['canonicalTitle'],
                'rating': entry['attributes']['averageRating'],
                'is_nsfw': entry['attributes']['nsfw'],
                'synopsis': entry['attributes']['synopsis'],
                'categories': {included_category['id']: {'id': included_category['id'], 'title': included_category['attributes']['title']} for included_category in entry.get('included', [])}
            }
        ]

    def _get_anime_batch(self, page_limit, page_offset):
        response_body = self._get_cache_or_fetch(f'{ANIME_ENDPOINT}',
                                                 params={'page[limit]': page_limit, 'page[offset]': page_offset},
                                                 headers=self._get_auth_header())
        return [
            {
                'id': entry['id'],
                'title': entry['attributes']['canonicalTitle'],
                'rating': entry['attributes']['averageRating'],
                'is_nsfw': entry['attributes']['nsfw'],
                'synopsis': entry['attributes']['synopsis']
            } for entry in response_body['data']
        ]

    def _get_category(self, category_id):
        response_body = self._get_cache_or_fetch(f'{CATEGORIES_ENDPOINT}/{category_id}', headers=self._get_auth_header())
        attributes = response_body['attributes']
        return {
            'id': response_body['data']['id'],
            'description': attributes['description'],
            'title': attributes['title']
        }

    def _get_auth_header(self):
        return {'Authorization': f'Bearer {self.auth["access_token"]}'}

    @staticmethod
    def _get_cache_or_fetch(path, use_cache=True, **kwargs):
        page_offset = kwargs.get('params', {}).get('page[offset]', 0)
        page_limit = kwargs.get('params', {}).get('page[limit]', 0)
        local_path = f'{CACHE_LOCATION}{path}_{page_offset}-{page_limit}.json'
        if use_cache and os.path.exists(local_path) and os.path.isfile(local_path):
            with open(local_path, 'r') as cached_json:
                response_body = json.load(cached_json)
                if response_body.get('errors'):
                    # pp.pprint(response.json())
                    raise IOError(f"Resource not found at {path}")
                return response_body
        else:
            response = requests.get(f'{KITSU_API_BASE_URL}{path}', **kwargs)
            if not os.path.exists(os.path.dirname(local_path)):
                try:
                    os.makedirs(os.path.dirname(local_path))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            with open(local_path, 'w+') as cached_json:
                json.dump(response.json(), cached_json)
                if response.status_code == 404:
                    # pp.pprint(response.json())
                    raise IOError(f"Resource not found at {path}")
                return response.json()
