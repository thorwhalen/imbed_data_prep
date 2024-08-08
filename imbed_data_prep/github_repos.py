"""Data prep for Github Repositories data.

Data source:
- [GitHub Public Repository Metadata](https://www.kaggle.com/datasets/pelmers/github-repository-metadata-with-5-stars?resource=download) 
- ([Dropbox link to parquet file](https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=0))

"""

from functools import partial
from io import BytesIO
import os
from dataclasses import dataclass, KW_ONLY
from typing import Sequence, Callable

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
    kmeans_cluster_indices,
    CacheSpec,
)
from imbed.base import extension_base_wrap
from lkj import clog, print_with_timestamp, log_calls as _log_calls

import pandas as pd
import numpy as np


raw_data_url = 'https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=1'

data_name = 'github_repos'
raw_data_name = 'github_repos.parquet'

# TODO: Use config2py tools to use and save default values
# TODO: Use config2py tools to include a message containing the default values
_DFLT_CACHE_DIR = saves_join(data_name)
DFLT_CACHE_DIR = get_config('GITHUB_REPOS_CACHE_DIR', default=_DFLT_CACHE_DIR)
_DFLT_RAW_DATA_FILEPATH = os.path.join(DFLT_CACHE_DIR, raw_data_name)
DFLT_RAW_DATA_FILEPATH = get_config(
    'GITHUB_REPOS_RAW_DATA_FILEPATH', default=_DFLT_RAW_DATA_FILEPATH
)
DFLT_N_CLUSTERS = (5, 8, 13, 21, 34)

# get_raw_data = partial(
#     url_to_file_download, raw_data_url, filepath=DFLT_RAW_DATA_FILEPATH, overwrite=False
# )

# get_base_data = Pipe(get_raw_data, BytesIO, pd.read_parquet)

# Have a separate config object? Like this:
# @dataclass
# class GithubReposConfig:
#     _: KW_ONLY
#     raw_data_url: str = (
#         'https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=1'
#     )
#     data_name = 'github_repos'
#     raw_data_name = 'github_repos.parquet'


# TODO: _cache_this doesn't work well with partial
# cache_this = partial(_cache_this, cache='cache')
cache_this = _cache_this

log_calls = _log_calls(
    logger=print_with_timestamp,
    log_condition=partial(_log_calls.instance_flag_is_set, flag_attr='verbose'),
)


@dataclass
class GithubReposData:
    _: KW_ONLY
    raw_data_src: str = (
        'https://www.dropbox.com/s/kokiypcm2ylx4an/github-repos.parquet?dl=1'
    )
    cache: CacheSpec = DFLT_CACHE_DIR
    raw_data_local_path: str = DFLT_RAW_DATA_FILEPATH
    planar_compute_chk_size: int = 1000
    embeddings_column: str = 'embedding'
    text_segments_column: str = 'text'
    planar_embeddings_func: PlanarEmbeddingSpec = 'ncvis'
    drop_duplicates: bool = True
    n_clusters: Sequence[int] = DFLT_N_CLUSTERS
    mk_cluster_learner: Callable = kmeans_cluster_indices
    verbose: int = 1

    def __post_init__(self):
        self.raw_data_local_path = ensure_fullpath(
            self.raw_data_local_path, conditional_rootdir=self.cache
        )
        self.cache = extension_base_wrap(ensure_cache(self.cache))
        self.log = clog(self.verbose)

        # TODO: Why do I need to do this?
        self.mk_cluster_learner = staticmethod(self.mk_cluster_learner)

    @property
    @log_calls
    def _raw_data_bytes(self):
        r = url_to_file_download(
            self.raw_data_src,
            filepath=self.raw_data_local_path,
            overwrite=False,
            url_egress=key_egress_print_downloading_message_with_size,
            return_func=return_contents,
        )
        return r

    @cache_this(cache=None)
    @log_calls
    def raw_data(self):
        df = pd.read_parquet(BytesIO(self._raw_data_bytes))
        if self.drop_duplicates:
            n_rows_before = len(df)
            self.log(f"Dropping duplicate nameWithOwner (github stub)...")
            df = df.drop_duplicates(subset=['nameWithOwner'])
            n_rows_after = len(df)
            self.log(f"... Dropped {n_rows_before - n_rows_after} duplicates")
        assert df.nameWithOwner.is_unique, f"Duplicate nameWithOwner in df"
        df = df.set_index("nameWithOwner", drop=False)
        return df

    @property
    def text_segments(self):
        return self.raw_data[self.text_segments_column]

    @property
    def embeddings(self):
        return self.raw_data[self.embeddings_column]

    @property
    def embeddings_matrix(self):
        # Note: NOT the same as self.embeddings.to_numpy()!!
        return np.array(self.embeddings.to_list())

    segment_vectors = embeddings  # alias

    @cache_this(cache='cache', key=add_extension('.parquet'))
    @log_calls
    def planar_embeddings(self):
        self.log(
            "Computing and saving planar embeddings with {self.planar_embeddings_func}"
        )
        r = planar_embeddings_dict_to_df(planar_embeddings(self.segment_vectors))
        return r

    def data_with_planar_embeddings(self):
        # join self.raw_data with self.planar_embeddings over the indices
        return pd.merge(
            self.raw_data, self.planar_embeddings, left_index=True, right_index=True
        )

    @cache_this(cache='cache', key=add_extension('.parquet'))
    @log_calls
    def cluster_indices(self):
        """
        Make a dataframe containing kmeans cluster indices for the original embeddings.
        """
        self.log(f"Computing cluster indices for num of clusters: {self.n_clusters}")

        def compute_cluster_indices():
            for n_clusters in self.n_clusters:
                yield self.mk_cluster_learner(
                    self.embeddings_matrix, n_clusters=n_clusters
                )

        cluster_columns = (
            f"clusters_{n_clusters:02.0f}" for n_clusters in self.n_clusters
        )

        r = pd.DataFrame(
            dict(zip(cluster_columns, compute_cluster_indices())),
            index=self.raw_data.index,
        )
        return r
