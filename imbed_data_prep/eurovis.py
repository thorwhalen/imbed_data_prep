"""Data prep for Eurovis data."""

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


import pandas as pd
import numpy as np


data_name = 'eurovis'

# raw_data_name = 'github_repos.parquet'

# TODO: Use config2py tools to use and save default values
# TODO: Use config2py tools to include a message containing the default values
_DFLT_CACHE_DIR = saves_join(data_name)

DFLT_CACHE_DIR = os.environ.get('EUROVIS_CACHE_DIR', default=_DFLT_CACHE_DIR)

DFLT_N_CLUSTERS = (5, 8, 13, 21, 34)


# TODO: _cache_this doesn't work well with partial
# cache_this = partial(_cache_this, cache='cache')
cache_this = partial(_cache_this, cache='saves', key=add_extension('parquet'))

import oa
from functools import lru_cache
from dataclasses import dataclass, KW_ONLY
from imbed.base import (
    LocalSavesMixin,
    DFLT_SAVES_DIR,
    add_token_info_to_df,
    DFLT_EMBEDDING_MODEL,
)
from imbed.data_prep import ImbedArtifactsMixin
from tabled import expand_rows, expand_columns
from dol.tools import cache_property_method


# TODO: Move embed_segments_one_by_one to reusables (e.g. imbed)
from typing import Mapping, Generator

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


@dataclass
class EurovisDacc(LocalSavesMixin, ImbedArtifactsMixin):
    name: Optional[str] = data_name
    _: KW_ONLY
    saves_dir: str = os.path.join(DFLT_SAVES_DIR, data_name)
    verbose: int = 1
    model: str = DFLT_EMBEDDING_MODEL

    @cache_this(key=add_extension('csv'))
    def raw_data(self):
        """The purpose of this function is to return the data that the user would
        have manually place in the saved_dir, so the method should never actually
        be called."""
        raise FileNotFoundError(
            "You need to place a file called `raw_data.csv` in the directory: "
            f"{self.saves_dir}"
        )

    @cache_this(pre_cache=True)
    def embeddable(self):
        # remove rows with missing title or abstract
        df = self.raw_data.dropna(subset=['title', 'abstract'])
        ## Add some util columns
        # contatenate title and abstract to make segment
        df['segment'] = '##' + df['title'] + '\n\n' + df.get('abstract')

        assert not any(df.segment.isna()), "some segments are NaN"

        # apply oa.num_tokens to segment to get n_tokens
        df['n_tokens'] = df['segment'].apply(oa.num_tokens)
        max_tokens = oa.util.embeddings_models[oa.base.DFLT_EMBEDDINGS_MODEL][
            'max_input'
        ]
        assert df['n_tokens'].max() <= max_tokens, "some segments exceed max tokens"

        df.set_index('doi', inplace=True)
        df.index.name = 'id_'
        return df

    # Replace by more scalable default
    @cache_this
    def embeddings_df(self):
        """WARNING: This could take A LONG TIME"""
        segments = self.segments()  # TODO: Should detect if property or method!
        vectors = dict(embed_segments_one_by_one(segments))
        df = pd.DataFrame(vectors).T
        df.index.name = 'id_'
        return df
    
    @cache_this
    def planar_embeddings(self):
        # TODO: Test this
        d = planar_embeddings(self.embeddings_df)
        # make it into a dataframe with the same index as the embeddings_df
        return planar_embeddings_dict_to_df(d)


    @property
    def segments(self):
        df = self.embeddable
        segments = dict(zip(df.doi, df.segment))
        assert len(segments) == len(df), "oops, duplicate DOIs"
        assert all(map(oa.text_is_valid, df.segment)), "some segments are invalid"

        return segments


    @cache_this
    def clusters_df(self):
        return super().clusters_df()
    

    @cache_this(cache='saves', key=add_extension('parquet'))
    def merged_artifacts(self):
        t = self.embeddable
        t = t.merge(self.planar_embeddings, left_index=True, right_index=True)
        t = t.merge(self.clusters_df, left_index=True, right_index=True)
        return t
