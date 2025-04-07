"""
Prepare wordnet words data.
"""

import os
from typing import Sized, Iterable
from functools import cached_property

import pandas as pd
from dol import cache_this
from lkj import clog
from i2 import Sig
from tabled import DfFiles

from nltk.corpus import wordnet as wn  # pip install nltk

DFLT_ROOTDIR = os.getenv('WORDNET_WORDS')


class WordsDacc:
    word_frequency_data_url = 'https://github.com/thorwhalen/content/raw/refs/heads/master/tables/csv/zip/english-word-frequency.csv.zip'

    def __init__(self, rootdir=DFLT_ROOTDIR, *, word_list=None, verbose=True):
        self.rootdir = rootdir
        if self.rootdir is None:
            raise ValueError("You need to specify a rootdir")
        self.df_files = DfFiles(rootdir)
        self.log = clog(verbose)
        if word_list is None:
            word_list = self._words_common_to_word_counts_and_wordnet()
        self.word_list = word_list

    def _words_common_to_word_counts_and_wordnet(self):
        """
        Take the intersection of the words that are indexed by wordnet (lemma names) and have a frequency count
        """
        return sorted(set(self.wordnet_words) & set(self.word_counts.index))

    @cache_this
    def words_set(self):
        return set(self.word_list)

    @cache_this  # to RAM
    def word_counts(self):
        # Note: The (..., keep_default_na=False, na_values=[]) is to avoid words "null" and "nan" being interpretted as NaN
        #    see https://www.skytowner.com/explore/preventing_strings_from_getting_parsed_as_nan_for_read_csv_in_pandas
        return pd.read_csv(
            self.word_frequency_data_url, keep_default_na=False, na_values=[]
        ).set_index('word')['count']

    @cache_this
    def wordnet_words(self):
        return sorted(list(wn.all_lemma_names()))

    @cache_this(cache='df_files', key='word_and_synset.parquet')
    def word_and_synset(self):
        def gen():
            for word in self.word_list:
                for synset in wn.synsets(word):
                    yield {
                        'word': word,
                        'synset': synset.name(),
                        'lemma': f"{synset.name()}.{word}",
                    }

        df = pd.DataFrame(gen())
        df = _deduplicate_lemmas_and_make_it_the_index(df)
        return df

    @cache_this(cache='df_files', key='wordnet_metadata.parquet')
    def wordnet_metadata(self):
        self.log('computing wordnet_metadata...')

        word_and_synset_df = self.word_and_synset

        synsets = word_and_synset_df.synset.unique()
        synsets_metadata = pd.DataFrame(synsets_metadatas(synsets))

        # translate pos to something user-friendly (replace pos with their full names)
        part_of_speech = {
            'n': 'noun',
            'v': 'verb',
            'a': 'adjective',
            's': 'adjective_satellite',
            'r': 'adverb',
        }
        synsets_metadata['pos'] = synsets_metadata['pos'].map(part_of_speech)

        # merge the two dataframes
        df = word_and_synset_df.merge(synsets_metadata, on='synset')

        df = _deduplicate_lemmas_and_make_it_the_index(df)
        return df

    @property
    def word_frequencies(self):
        t = self.word_counts
        t = t / t.sum()
        t.name = "frequency"
        return t

    @cache_this(cache='df_files', key='wordnet_feature_meta.parquet')
    def wordnet_feature_meta(self):
        t = self.wordnet_metadata[wordnet_feature_attr_names]
        t = self.word_frequencies.loc[self.word_list].to_frame()
        t = t.merge(self.umap_embeddings, left_on="word", right_index=True, how="left")
        t = t.merge(
            self.wordnet_metadata[['word'] + wordnet_feature_attr_names],
            right_on='word',
            left_index=True,
            # right_index=True,
            how="left",
        )
        non_wordnet_features = ['word', 'frequency']
        embedding_features = ['umap_x', 'umap_y']
        return t[non_wordnet_features + wordnet_feature_attr_names + embedding_features]

    @cache_this(cache='df_files', key='wordnet_collection_meta.parquet')
    def wordnet_collection_meta(self):
        return self.wordnet_metadata[wordnet_collection_attr_names]

    @cache_this(cache='df_files', key='words_embeddings.parquet')
    def words_embeddings(self):
        self.log('computing words_embeddings...')
        import oa  # pip install oa

        assert len(self.word_list) == len(set(self.word_list)), "Words not unique"
        word_embeddings = oa.embeddings(self.word_list)
        df = pd.DataFrame(
            index=self.word_list, data=map(lambda x: [x], word_embeddings)
        )
        df.columns = ['embedding']
        return df

    @cache_this(cache='df_files', key='umap_embeddings.parquet')
    def umap_embeddings(self):
        self.log('computing UMAP planar embeddings of words_embeddings...')

        import imbed

        umap_planar_embeddings = imbed.umap_2d_embeddings(
            self.words_embeddings.embedding
        )
        return imbed.planar_embeddings_dict_to_df(
            umap_planar_embeddings, x_col='umap_x', y_col='umap_y', index_name='word'
        )

    @cache_this
    def lemmas_of_synset(self):
        """
        Precompute synset_to_lemmas: mapping from synset to all lemmas that share that synset
        """
        from collections import defaultdict

        synset_to_lemmas = defaultdict(set)

        for lemma, row in self.word_and_synset.iterrows():
            syn = row['synset']
            synset_to_lemmas[syn].add(lemma)
        return synset_to_lemmas

    @cache_this
    def word_of_lemma(self):
        """
        Precompute synset_to_lemmas: mapping from synset to all lemmas that share that synset
        """

        # Precompute lemma_to_word: mapping from lemma to word (a lemma can be used as a key to find its word)

        # word_and_synset is a dataframe indexed by lemma, with columns: word, synset
        lemma_to_word = {}

        for lemma, row in self.word_and_synset.iterrows():
            lemma_to_word[lemma] = row['word']

        return lemma_to_word

    def lemma_graph_edges(self, adjacency):
        # For each source lemma in adjacency, find the target synsets.
        # For each target synset, gather its associated lemmas.
        # Yield a dict with "source" and "target" keys for each resulting lemma-to-lemma edge.

        # TODO: Can be optimized
        for source_lemma, target_synsets in adjacency.items():
            # target_synsets are synset identifiers
            for target_syn in target_synsets:
                for target_lemma in self.lemmas_of_synset.get(target_syn, []):
                    if (
                        source_lemma in self.word_of_lemma
                        and target_lemma in self.word_of_lemma
                    ):
                        yield {
                            "source": source_lemma,
                            "target": target_lemma,
                        }

    # @cache_this(cache='df_files', key='words_and_features.parquet')
    def words_and_features(self):
        t = self.word_counts
        word_frequencies = t / t.sum()
        word_frequencies.name = "frequency"

        t = self.wordnet_feature_meta.merge(
            word_frequencies, left_on="word", right_index=True, how="left"
        )
        t = t.merge(self.umap_embeddings, left_on="word", right_index=True, how="left")
        return t

    # @property
    # def synset_details(self):
    #     """
    #     Generate details for words using WordNet synsets, with simplification via getattr.

    #     Args:
    #         words (list): List of words to query in WordNet.

    #     Yields:
    #         dict: A dictionary of synset details for each word.
    #     """

    #     attributes_to_extract = self._non_string_iterable_columns

    #     for word in self.word_list:
    #         for synset in wn.synsets(word):
    #             synset_data = {
    #                 "word": word,
    #                 "synset": synset.name(),  # Synset identifier (e.g., "salt.n.03")
    #                 "definition": synset.definition(),  # Definition of the synset
    #                 "example": next(
    #                     iter(synset.examples()), ''
    #                 ),  # Example usage or empty string
    #                 "pos": synset.pos(),  # Part of speech
    #             }

    #             # Use getattr to dynamically populate lists for attributes
    #             for method_name in attributes_to_extract:
    #                 synset_data[method_name] = [
    #                     x.name() for x in getattr(synset, method_name)()
    #                 ]

    #             yield synset_data


def synsets_metadatas(synsets):
    """
    A generator that takes a list of words and yields details for each word and synset.

    Args:
        words (list): A list of words (strings) to query WordNet.

    Yields:
        dict: A dictionary with information about the word, synset, and associated features.
    """
    from nltk.corpus import wordnet as wn

    for synset in synsets:
        if isinstance(synset, str):
            synset = wn.synset(synset)

        d = {
            "synset": synset.name(),  # Synset identifier (e.g., "salt.n.03")
            "example": next(iter(synset.examples()), ''),
        }
        for attr_name in wordnet_feature_attr_names:
            d[attr_name] = getattr(synset, attr_name)()
        for attr_name in wordnet_collection_attr_names:
            val = getattr(synset, attr_name)()
            try:
                d[attr_name] = [synset.name() for synset in val]
            except Exception as e:
                raise print(f"{attr_name} failed. {val=}. {e=}")

        yield d


import pandas as pd


def _deduplicate_lemmas_and_make_it_the_index(df):
    # get rid of duplicate lemma names
    df['lemma'] = (
        df['synset'] + '.' + df['word']
    )  # note: didn't find an attribute for this
    df = df.drop_duplicates(
        subset='lemma'
    )  # drop duplicates (don't know why they're there, but they are)
    df = df.set_index('lemma')
    assert not any(
        df.index.duplicated()
    ), "There were some duplicates lemmas in the index"
    return df


# --------------------------------------------------------------------------------------
# Constants

from nltk.corpus.reader.wordnet import Lemma, Synset

_example_synset = wn.synsets('dog')[0]

_exclude_attr_names = (
    'mst',
    'acyclic_tree',
    'also_sees',
    'frame_ids',
    'max_depth',
    'min_depth',
    'offset',
    'lemmas',
    'examples',  # will be handled separately
    'hypernym_distances',  # is a set of tuples
    'hypernym_paths',  # a list of lists,
    'lemma_names',  # a list of strings
)


def _get_wordnet_attributes_example(exclude_attr_names=_exclude_attr_names):
    """
    Generate WordNet attributes for a synset.

    This is what was used to define wordnet_attrs, but the result was then copied to
    the module level for stability.

    Do

    ```
    wordnet_attrs_example = dict(_get_wordnet_attributes_example())
    ```

    to recompute the attributes.
    """
    exclude_attr_names = set(exclude_attr_names)
    for attr_name in dir(_example_synset):
        if not attr_name.startswith('_') and attr_name not in exclude_attr_names:
            attr = getattr(_example_synset, attr_name)
            if callable(attr) and len(Sig(attr).required_names) == 0:
                yield attr_name, attr()


wordnet_attrs_example = {
    'attributes': [],
    'causes': [],
    'definition': 'a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds',
    'entailments': [],
    'examples': ['the dog barked all night'],
    'hypernym_distances': {
        (Synset('animal.n.01'), 2),
        (Synset('animal.n.01'), 7),
        (Synset('canine.n.02'), 1),
        (Synset('carnivore.n.01'), 2),
        (Synset('chordate.n.01'), 6),
        (Synset('dog.n.01'), 0),
        (Synset('domestic_animal.n.01'), 1),
        (Synset('entity.n.01'), 8),
        (Synset('entity.n.01'), 13),
        (Synset('living_thing.n.01'), 4),
        (Synset('living_thing.n.01'), 9),
        (Synset('mammal.n.01'), 4),
        (Synset('object.n.01'), 6),
        (Synset('object.n.01'), 11),
        (Synset('organism.n.01'), 3),
        (Synset('organism.n.01'), 8),
        (Synset('physical_entity.n.01'), 7),
        (Synset('physical_entity.n.01'), 12),
        (Synset('placental.n.01'), 3),
        (Synset('vertebrate.n.01'), 5),
        (Synset('whole.n.02'), 5),
        (Synset('whole.n.02'), 10),
    },
    'hypernym_paths': [
        [
            Synset('entity.n.01'),
            Synset('physical_entity.n.01'),
            Synset('object.n.01'),
            Synset('whole.n.02'),
            Synset('living_thing.n.01'),
            Synset('organism.n.01'),
            Synset('animal.n.01'),
            Synset('domestic_animal.n.01'),
            Synset('dog.n.01'),
        ],
        [
            Synset('entity.n.01'),
            Synset('physical_entity.n.01'),
            Synset('object.n.01'),
            Synset('whole.n.02'),
            Synset('living_thing.n.01'),
            Synset('organism.n.01'),
            Synset('animal.n.01'),
            Synset('chordate.n.01'),
            Synset('vertebrate.n.01'),
            Synset('mammal.n.01'),
            Synset('placental.n.01'),
            Synset('carnivore.n.01'),
            Synset('canine.n.02'),
            Synset('dog.n.01'),
        ],
    ],
    'hypernyms': [Synset('domestic_animal.n.01'), Synset('canine.n.02')],
    'hyponyms': [
        Synset('pooch.n.01'),
        Synset('pug.n.01'),
        Synset('griffon.n.02'),
        Synset('spitz.n.01'),
        Synset('toy_dog.n.01'),
        Synset('basenji.n.01'),
        Synset('great_pyrenees.n.01'),
        Synset('working_dog.n.01'),
        Synset('hunting_dog.n.01'),
        Synset('poodle.n.01'),
        Synset('mexican_hairless.n.01'),
        Synset('puppy.n.01'),
        Synset('leonberg.n.01'),
        Synset('newfoundland.n.01'),
        Synset('corgi.n.01'),
        Synset('dalmatian.n.02'),
        Synset('cur.n.01'),
        Synset('lapdog.n.01'),
    ],
    'in_region_domains': [],
    'in_topic_domains': [],
    'in_usage_domains': [],
    'instance_hypernyms': [],
    'instance_hyponyms': [],
    'lemma_names': ['dog', 'domestic_dog', 'Canis_familiaris'],
    'lexname': 'noun.animal',
    'member_holonyms': [Synset('pack.n.06'), Synset('canis.n.01')],
    'member_meronyms': [],
    'name': 'dog.n.01',
    'part_holonyms': [],
    'part_meronyms': [Synset('flag.n.07')],
    'pos': 'n',
    'region_domains': [],
    'root_hypernyms': [Synset('entity.n.01')],
    'similar_tos': [],
    'substance_holonyms': [],
    'substance_meronyms': [],
    'topic_domains': [],
    'usage_domains': [],
    'verb_groups': [],
}


wordnet_attrs_example = dict(_get_wordnet_attributes_example())
wordnet_attr_names = sorted(wordnet_attrs_example)
wordnet_collection_attr_names = sorted(
    [
        name
        for name in wordnet_attr_names
        if not isinstance(wordnet_attrs_example[name], str)
        and isinstance(wordnet_attrs_example[name], Sized)
    ]
)
wordnet_feature_attr_names = sorted(
    set(wordnet_attr_names) - set(wordnet_collection_attr_names)
)
