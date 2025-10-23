"""Data prep for Eurovis data."""

from functools import partial
import os
from dataclasses import dataclass, KW_ONLY
from typing import Optional

from dol import cache_this as _cache_this, add_extension
from imbed.util import (
    saves_join,
    planar_embeddings,
    planar_embeddings_dict_to_df,
)


import pandas as pd


data_name = 'twitter_sentiment'

# raw_data_name = 'github_repos.parquet'

# TODO: Use config2py tools to use and save default values
# TODO: Use config2py tools to include a message containing the default values
_DFLT_CACHE_DIR = saves_join(data_name)

DFLT_CACHE_DIR = os.environ.get('TWITTER_SENTIMENT_DATA_DIR', default=_DFLT_CACHE_DIR)

DFLT_N_CLUSTERS = (5, 8, 13, 21, 34)


# TODO: _cache_this doesn't work well with partial
# cache_this = partial(_cache_this, cache='cache')
cache_this = partial(_cache_this, cache='saves', key=add_extension('parquet'))

import oa
from dataclasses import dataclass, KW_ONLY
from imbed.base import (
    LocalSavesMixin,
    DFLT_SAVES_DIR,
    DFLT_EMBEDDING_MODEL,
)
from imbed.data_prep import ImbedArtifactsMixin


# TODO: Move embed_segments_one_by_one to reusables (e.g. imbed)
from collections.abc import Mapping, Generator

KeyVectorPairs = Generator[tuple[str, list[float]], None, None]


def embed_segments_one_by_one(segments: Mapping[str, str]) -> KeyVectorPairs:
    print(f"This could take a LONG TIME!!!")
    assert isinstance(segments, dict)
    for i, (key, segment) in enumerate(segments.items()):
        try:
            vector = oa.embeddings(segment)
            yield key, vector
        except Exception as e:
            print(f"Error processing ({i=}) {key}: {e}")
            # continue


import pandas as pd
from haggle import KaggleDatasets
from imbed import extension_based_wrap


@dataclass
class Dacc(LocalSavesMixin, ImbedArtifactsMixin):
    name: str | None = data_name
    _: KW_ONLY
    saves_dir: str = os.path.join(DFLT_SAVES_DIR, data_name)
    verbose: int = 1
    model: str = DFLT_EMBEDDING_MODEL

    column_info = {
        "target": "the polarity of the tweet (e.g. 0 = negative, 2 = neutral, 4 = positive)",
        "id_": "The id of the tweet (e.g. 2087)",
        "date": "the date of the tweet (e.g. Sat May 16 23:58:44 UTC 2009)",
        "flag": "The query (lyx). If there is no query, then this value is NO_QUERY.",
        "user": "the user that tweeted (e.g. robotickilldozr)",
        "text": "the text of the tweet (e.g. Lyx is cool)",
    }

    @cache_this(cache=None)
    def raw_data(self):
        """The purpose of this function is to return the data that the user would
        have manually place in the saved_dir, so the method should never actually
        be called."""

        kaggle = KaggleDatasets()

        raw_data_store = extension_based_wrap(kaggle['kazanova/sentiment140'])
        list(raw_data_store)
        df = raw_data_store['training.1600000.processed.noemoticon.csv']

        # there was no header in the csv file, so we need to specify the column names
        columns = list(
            self.column_info.keys()
        )  # (was extracted from https://www.kaggle.com/datasets/kazanova/sentiment140)
        # we need to move the current columns of df to be the first row,
        #  and then set the columns to be the columns we want
        df = pd.DataFrame([df.columns] + df.values.tolist(), columns=columns)

        return df

    # @cache_this(pre_cache=True)
    # def embeddable(self):
    #     # remove rows with missing title or abstract
    #     df = self.raw_data.dropna(subset=['title', 'abstract'])
    #     ## Add some util columns
    #     # contatenate title and abstract to make segment
    #     df['segment'] = '##' + df['title'] + '\n\n' + df.get('abstract')

    #     assert not any(df.segment.isna()), "some segments are NaN"

    #     # apply oa.num_tokens to segment to get n_tokens
    #     df['n_tokens'] = df['segment'].apply(oa.num_tokens)
    #     max_tokens = oa.util.embeddings_models[oa.base.DFLT_EMBEDDINGS_MODEL][
    #         'max_input'
    #     ]
    #     assert df['n_tokens'].max() <= max_tokens, "some segments exceed max tokens"

    #     df.set_index('doi', drop=False, inplace=True)
    #     df.index.name = 'id_'
    #     return df

    # # Replace by more scalable default
    # @cache_this
    # def embeddings_df(self):
    #     """WARNING: This could take A LONG TIME"""
    #     segments = self.segments()  # TODO: Should detect if property or method!
    #     vectors = dict(embed_segments_one_by_one(segments))
    #     df = pd.DataFrame(vectors).T
    #     df.index.name = 'id_'
    #     return df

    # @cache_this
    # def planar_embeddings(self):
    #     # TODO: Test this
    #     d = planar_embeddings(self.embeddings_df)
    #     # make it into a dataframe with the same index as the embeddings_df
    #     return planar_embeddings_dict_to_df(d)

    # @property
    # def segments(self):
    #     df = self.embeddable
    #     segments = dict(zip(df.index.values, df.segment))
    #     assert len(segments) == len(df), "oops, duplicate DOIs"
    #     assert all(
    #         map(oa.text_is_valid, df.segment.values)
    #     ), "some segments are invalid"

    #     return segments

    # @cache_this
    # def clusters_df(self):
    #     return super().clusters_df()

    # @cache_this(cache='saves', key=add_extension('parquet'))
    # def merged_artifacts(self):
    #     t = self.embeddable
    #     t = t.merge(self.planar_embeddings, left_index=True, right_index=True)
    #     t = t.merge(self.clusters_df, left_index=True, right_index=True)
    #     return t

    # @cache_this
    # def testing123(self):
    #     return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
