import errno
import os
import sys

import requests
import pprint
import json

from api_client import AnimeApiClient

ANILIST_API_URL = 'https://graphql.anilist.co'

CACHE_LOCATION = './.cache/anilist'

anime_graphQL_query = """
query ($page: Int, $perPage: Int){
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
      perPage
    }
    media {
      type
      id
      title {
        english
        romaji
      }
      description
      isAdult
      averageScore
      genres
    }
  }
}
"""

pp = pprint.PrettyPrinter(compact=True)


class Anilist(AnimeApiClient):

    def get_anime_range(self, begin, end, batch_size=50):
        anime_list = []
        end = end if end else sys.maxsize
        begin_batch = begin // batch_size
        end_batch = end // batch_size
        for i in range(begin_batch, end_batch):
            if (i - begin_batch) % (1000 // batch_size) == 0:
                print(f'Currently querying\t\ti={i * batch_size}\t\tbegin={begin}\t\tend={end}')
            try:
                batch, has_next_page = self._get_anime_batch(batch_size, i)
                anime_list += batch
                if not has_next_page:
                    break
            except IOError as ioe:
                print(ioe, file=sys.stderr)
        return anime_list

    def _get_anime_batch(self, page_limit, page_offset):
        response_body = self._get_cache_or_fetch('/anime', query=anime_graphQL_query, variables={'perPage': page_limit, 'page': page_offset})
        return [
                   {
                       'id': entry['id'],
                       'title': entry['title']['romaji'],
                       'rating': entry['averageScore'],
                       'is_nsfw': entry['isAdult'],
                       'synopsis': entry['description'],
                       'tags': entry['genres'],
                   } for entry in response_body['data']['Page']['media']
               ], response_body['data']['Page']['pageInfo']['hasNextPage']

    @staticmethod
    def _get_cache_or_fetch(path, query, variables, use_cache=True, **kwargs):
        page_offset = variables['page']
        page_limit = variables['perPage']
        local_path = f'{CACHE_LOCATION}{path}_{page_offset}-{page_limit}.json'
        if use_cache and os.path.exists(local_path) and os.path.isfile(local_path):
            with open(local_path, 'r') as cached_json:
                response_body = json.load(cached_json)
                return response_body
        else:
            response = requests.post(f'{ANILIST_API_URL}', json={'query': query, 'variables': variables})
            if use_cache:
                if not os.path.exists(os.path.dirname(local_path)):
                    try:
                        os.makedirs(os.path.dirname(local_path))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                with open(local_path, 'w+') as cached_json:
                    json.dump(response.json(), cached_json)
            return response.json()
