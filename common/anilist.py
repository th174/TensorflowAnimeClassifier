import json
import os
import pprint
import sys

import requests
from tensorflow import io

from common.client import AnimeApiClient

ANILIST_API_URL = 'https://graphql.anilist.co'

CACHE_LOCATION = '{}/cache'.format(os.path.dirname(__file__))

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
    def __init__(self, cache_location=CACHE_LOCATION):
        self.anime_list = []
        self.cache_location = cache_location

    def get_anime_range(self, begin, end, batch_size=50):
        self.anime_list = []
        end = end if end else sys.maxsize
        begin_batch = begin // batch_size
        end_batch = end // batch_size
        print("Saving files to {}/anilist".format(self.cache_location), flush=True)
        for i in range(begin_batch, end_batch):
            if (i - begin_batch) % (1000 // batch_size) == 0:
                print('Currently querying\t\ti={}\t\tbegin={}\t\tend={}'.format(i * batch_size, begin, end), flush=True)
            try:
                batch, has_next_page = self._get_anime_batch(batch_size, i)
                self.anime_list += batch
                if not has_next_page:
                    break
            except IOError as ioe:
                print(ioe, file=sys.stderr, flush=True)
        return self.anime_list

    def _get_anime_batch(self, page_limit, page_offset):
        response_body = self._get_cache_or_fetch('/anime', query=anime_graphQL_query, variables={'perPage': page_limit, 'page': page_offset})
        try:
            return [
                       {
                           'id': entry['id'],
                           'title': entry['title']['romaji'],
                           'rating': entry['averageScore'],
                           'is_nsfw': entry['isAdult'],
                           'synopsis': entry['description'],
                           'tags': entry['genres'],
                       }
                       for entry in response_body['data']['Page']['media']
                   ], response_body['data']['Page']['pageInfo']['hasNextPage']
        except:
            print(response_body, file=sys.stderr, flush=True)

    def _get_cache_or_fetch(self, path, query, variables, use_cache=True):
        page_offset = variables['page']
        page_limit = variables['perPage']
        local_path = '{}/anilist{}_{:04d}-{}.json'.format(self.cache_location, path, page_offset, page_offset * page_limit)
        if use_cache and io.gfile.exists(local_path):
            with io.gfile.GFile(local_path, 'r') as cached_json:
                response_body = json.load(cached_json)
                print(local_path + "\n" + response_body, file=sys.stderr, flush=True)
                return response_body
        elif not self.cache_location.startswith("gs://"):
            response = requests.post('{}'.format(ANILIST_API_URL), json={'query': query, 'variables': variables})
            if use_cache:
                io.gfile.makedirs(os.path.dirname(local_path))
                with io.gfile.GFile(local_path, 'w+') as cached_json:
                    json.dump(response.json(), cached_json)
            return response.json()
        else:
            raise Exception("Path not found:\n{}".format(local_path))

    def __str__(self):
        return "Anilist"
