"""Data prep for Github Repositories data.

Data source:
- [GitHub Public Repository Metadata](https://www.kaggle.com/datasets/pelmers/github-repository-metadata-with-5-stars?resource=download) 
- ([Dropbox link to parquet file](https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=0))

"""

from functools import partial
from io import BytesIO
import os
from dataclasses import dataclass, KW_ONLY
from typing import Sequence, Callable, Optional

from dol import cache_this as _cache_this, Files, Pipe
from graze import url_to_file_download
from graze.base import (
    return_filepath,
    return_contents,
    key_egress_print_downloading_message_with_size,
)
from dol.tools import cache_property_method
from imbed.util import (
    saves_join,
    get_config,
    planar_embeddings,
    PlanarEmbeddingSpec,
    planar_embeddings_dict_to_df,
    DFLT_PLANAR_EMBEDDING_KIND,
    ensure_fullpath,
    ensure_cache,
    add_extension,
    CacheSpec,
    log_calls,
    log_method_calls,
    counts,
)
from imbed.data_prep import kmeans_cluster_indices
from imbed.base import extension_based_wrap

import pandas as pd
import numpy as np


data_name = 'wildchat'
huggingface_data_stub = "allenai/WildChat-1M"
# raw_data_name = 'github_repos.parquet'

# TODO: Use config2py tools to use and save default values
# TODO: Use config2py tools to include a message containing the default values
_DFLT_CACHE_DIR = saves_join(data_name)

DFLT_CACHE_DIR = os.environ.get('WILDCHAT_CACHE_DIR', default=_DFLT_CACHE_DIR)
# # For a more flexible configs getter (that will ask user for missing keys):
# DFLT_CACHE_DIR = get_config('WILDCHAT_CACHE_DIR', default=_DFLT_CACHE_DIR)

# _DFLT_RAW_DATA_FILEPATH = os.path.join(DFLT_CACHE_DIR, raw_data_name)
# DFLT_RAW_DATA_FILEPATH = get_config(
#     'GITHUB_REPOS_RAW_DATA_FILEPATH', default=_DFLT_RAW_DATA_FILEPATH
# )
DFLT_N_CLUSTERS = (5, 8, 13, 21, 34)


# TODO: _cache_this doesn't work well with partial
# cache_this = partial(_cache_this, cache='cache')
cache_this = _cache_this

import oa
from dataclasses import dataclass, KW_ONLY
from imbed.base import (
    HugfaceDaccBase,
    DFLT_SAVES_DIR,
    add_token_info_to_df,
    DFLT_EMBEDDING_MODEL,
)
from tabled import expand_rows, expand_columns


@dataclass
class WildchatDacc(HugfaceDaccBase):
    huggingface_data_stub: str = huggingface_data_stub
    name: Optional[str] = data_name
    _: KW_ONLY
    saves_dir: Optional[str] = None
    root_saves_dir: str = DFLT_SAVES_DIR
    verbose: int = 1
    model: str = DFLT_EMBEDDING_MODEL

    @cache_this(cache='saves', key=add_extension('parquet'))
    @log_method_calls
    def expanded_train(self):
        return expand_wildchat_data(self.train_data)

    @property
    def expanded_en(self):
        return self.expanded_train[self.expanded_train.language == 'English']

    @property
    def language_conversation_counts(self):
        return counts(self.train_data.language)

    @property
    def language_turn_counts(self):
        return counts(self.expanded_train.language)

    # @cache_this(cache='saves', key=add_extension('.parquet'))
    # @log_method_calls
    @cache_this(cache='saves', key=add_extension('.parquet'))
    @log_method_calls
    def embeddable_df(self):
        df = self.expanded_train
        df['id_'] = df.index.values
        df['segment_length'] = df['conversation.content'].apply(len)
        df = add_token_info_to_df(
            df,
            segments_col='conversation.content',
            model=self.model,
        )  # takes 10mn
        # then keep only those conversations (grouped by hash) that have all valid segments
        return df.groupby('conversation_hash').filter(
            lambda x: x['segment_is_valid'].all()
        )


def expand_wildchat_data(df):
    """
    Expand columns and rows of raw wildchat data, to get a more flattened dataframe.

    The raw data contains several columns that have nested data. More specifically,
    the columns -- 'conversation', 'openai_moderation', and 'detoxify_moderation' --
    which contain (aligned) lists of dicts, where each dict of a list corresponds to
    a "part" of the conversation conversation: Either the "user", or the "assistant"
    role -- the size of this list should be the number of "turns" times two (user part,
    followed with assistant response).

    So we expand the lists, thus expanding the rows to contain exactly a part each.
    Then we expand the dicts, thus expanding the colums, to contain a field of the dict
    each.
    Finally, there's one more level of nested dicts in the
    'openai_moderation.categories' and 'openai_moderation.category_scores' columns,
    which we also flatten by expanding columns.
    Note that before we do this, we drop rows that have null openai_moderation values.
    There were 16996 such rows in the (row-expanded) raw data.

    """
    # for each (aligned) item of conversation and moderation, make a new row
    df = expand_rows(df, ['conversation', 'openai_moderation', 'detoxify_moderation'])
    # These items are dicts, so expand their fields into columns
    df = expand_columns(
        df, ['conversation', 'openai_moderation', 'detoxify_moderation']
    )
    # openai_moderation has yet one more level of nesting so flatten that too
    openai_moderation_cols = [
        'openai_moderation.categories',
        'openai_moderation.category_scores',
    ]
    #   some 16996 rows have null openai_moderation values: drop them before expanding
    df.dropna(subset=openai_moderation_cols, inplace=True)
    df = expand_columns(df, openai_moderation_cols)
    # remove all columns that have a / in them (because they have duplicates)
    df = df[[c for c in df.columns if '/' not in c]]
    return df


# @dataclass
# class GithubReposData:
#     _: KW_ONLY
#     raw_data_src: str = (
#         'https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=1'
#     )
#     cache: CacheSpec = DFLT_CACHE_DIR
#     raw_data_local_path: str = DFLT_RAW_DATA_FILEPATH
#     planar_compute_chk_size: int = 1000
#     embeddings_column: str = 'embedding'
#     text_segments_column: str = 'text'
#     planar_embeddings_func: PlanarEmbeddingSpec = 'ncvis'
#     drop_duplicates: bool = True
#     n_clusters: Sequence[int] = DFLT_N_CLUSTERS
#     mk_cluster_learner: Callable = kmeans_cluster_indices
#     verbose: int = 1

#     def __post_init__(self):
#         self.raw_data_local_path = ensure_fullpath(
#             self.raw_data_local_path, conditional_rootdir=self.cache
#         )
#         self.cache = extension_based_wrap(ensure_cache(self.cache))
#         self.log = clog(self.verbose)

#         # TODO: Why do I need to do this?
#         self.mk_cluster_learner = staticmethod(self.mk_cluster_learner)

#     @property
#     @log_calls
#     def _raw_data_bytes(self):
#         r = url_to_file_download(
#             self.raw_data_src,
#             filepath=self.raw_data_local_path,
#             overwrite=False,
#             url_egress=key_egress_print_downloading_message_with_size,
#             return_func=return_contents,
#         )
#         return r

#     @cache_this(cache=None)
#     @log_calls
#     def raw_data(self):
#         df = pd.read_parquet(BytesIO(self._raw_data_bytes))
#         if self.drop_duplicates:
#             n_rows_before = len(df)
#             self.log(f"Dropping duplicate nameWithOwner (github stub)...")
#             df = df.drop_duplicates(subset=['nameWithOwner'])
#             n_rows_after = len(df)
#             self.log(f"... Dropped {n_rows_before - n_rows_after} duplicates")
#         assert df.nameWithOwner.is_unique, f"Duplicate nameWithOwner in df"
#         df = df.set_index("nameWithOwner", drop=False)
#         return df

#     @property
#     def text_segments(self):
#         return self.raw_data[self.text_segments_column]

#     @property
#     def embeddings(self):
#         return self.raw_data[self.embeddings_column]

#     @property
#     def embeddings_matrix(self):
#         # Note: NOT the same as self.embeddings.to_numpy()!!
#         return np.array(self.embeddings.to_list())

#     segment_vectors = embeddings  # alias

#     @cache_this(cache='cache', key=add_extension('.parquet'))
#     @log_calls
#     def planar_embeddings(self):
#         self.log(
#             "Computing and saving planar embeddings with {self.planar_embeddings_func}"
#         )
#         r = planar_embeddings_dict_to_df(planar_embeddings(self.segment_vectors))
#         return r

#     def data_with_planar_embeddings(self):
#         # join self.raw_data with self.planar_embeddings over the indices
#         return pd.merge(
#             self.raw_data, self.planar_embeddings, left_index=True, right_index=True
#         )

#     @cache_this(cache='cache', key=add_extension('.parquet'))
#     @log_calls
#     def cluster_indices(self):
#         """
#         Make a dataframe containing kmeans cluster indices for the original embeddings.
#         """
#         self.log(f"Computing cluster indices for num of clusters: {self.n_clusters}")

#         def compute_cluster_indices():
#             for n_clusters in self.n_clusters:
#                 yield self.mk_cluster_learner(
#                     self.embeddings_matrix, n_clusters=n_clusters
#                 )

#         cluster_columns = (
#             f"clusters_{n_clusters:02.0f}" for n_clusters in self.n_clusters
#         )

#         r = pd.DataFrame(
#             dict(zip(cluster_columns, compute_cluster_indices())),
#             index=self.raw_data.index,
#         )
#         return r
