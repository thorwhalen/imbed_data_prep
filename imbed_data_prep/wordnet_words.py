"""
Prepare wordnet words data.

The steps are:
* Get a list of most frequent (English) words
* Get embeddings for each of these words
* Get planar projections for these embeddings
* Link the words in various ways (i.e make the link data)

## Architecture: Synset-Based Indexing

This module uses **synsets as the primary index** for metadata and link data, avoiding the
edge explosion that comes from replicating concept→concept edges across all lemma combinations.

In WordNet, a **synset** (short for "synonym set") is a group of words that share the same
meaning or concept. Think of it as a cluster of synonyms that can be used interchangeably
in certain contexts without changing the overall meaning. For example, the words "happy,"
"joyful," and "elated" might belong to the same synset because they convey similar emotions.

### Key Data Structures

- `wordnet_metadata`: Synset-indexed DataFrame with a 'lemmas' column (list of words)
- `word_and_synset`: Bipartite mapping between words and synsets
- `synset_definition_links`: Synset→synset edges (concept graph backbone)
- `word_indexed_metadata`: Word-indexed view with one representative synset per word

This design keeps the concept graph clean and scalable, while still supporting word-level
queries through the bipartite word↔synset layer.

The synset relationships are as follows:

* **attributes**: These are qualities or characteristics associated with a synset. *Example*: For the synset representing "banana," an attribute might be "yellow."
* **causes**: This relationship indicates that one synset brings about or results in another. *Example*: "Tickling" (synset) causes "laughter" (synset).
* **entailments**: Primarily used for verbs, this relationship means that one action logically necessitates another. *Example*: If someone is "snoring," it entails that they are "sleeping."
* **hypernyms**: A hypernym is a more general term that encompasses more specific instances. *Example*: "Vehicle" is a hypernym of "car."
* **hyponyms**: A hyponym is a more specific term within a broader category. *Example*: "Poodle" is a hyponym of "dog."
* **in_region_domains**: This denotes the regional usage of a synset, indicating where a term is commonly used. *Example*: The term "biscuit" in British English refers to what Americans call a "cookie."
* **in_topic_domains**: This shows the subject area or field to which a synset belongs. *Example*: The term "quantum" belongs to the domain of physics.
* **in_usage_domains**: This indicates the context or manner in which a term is used. *Example*: "LOL" is used in informal, internet communication.
* **instance_hypernyms**: This relationship links a specific instance to its general category. *Example*: "Einstein" is an instance of the hypernym "physicist."
* **instance_hyponyms**: This connects a general category to its specific instances. *Example*: "Physicist" has instance hyponyms like "Einstein" and "Newton."
* **member_holonyms**: This indicates the whole to which a member belongs. *Example*: A "tree" is a member of the holonym "forest."
* **member_meronyms**: This shows the members that constitute a collective whole. *Example*: "Player" is a member meronym of "team."
* **part_holonyms**: This denotes the whole object that a part belongs to. *Example*: A "wheel" is part of the holonym "car."
* **part_meronyms**: This indicates the parts that make up a whole object. *Example*: "Keyboard" is a part meronym of "computer."
* **region_domains**: This specifies the geographical area where a term is used. *Example*: "G'day" is used in the region domain of Australia.
* **root_hypernyms**: This refers to the most general term in a hierarchy. *Example*: For "poodle," the root hypernym might be "entity."
* **similar_tos**: This indicates synsets that are similar in meaning. *Example*: "Big" is similar to "large."
* **substance_holonyms**: This shows the whole that a substance is part of. *Example*: "Flour" is a substance holonym of "bread."
* **substance_meronyms**: This indicates the substances that make up a whole. *Example*: "Alcohol" is a substance meronym of "wine."
* **topic_domains**: This denotes the subject area a term is associated with. *Example*: "Molecule" belongs to the topic domain of chemistry.
* **usage_domains**: This specifies the context in which a term is appropriately used. *Example*: "Thou" is used in archaic or poetic contexts.
* **verb_groups**: This links verbs that are similar in meaning or usage. *Example*: "Run" and "jog" might be in the same verb group.

"""

import os
import re
from collections.abc import Sized, Iterable
from functools import cached_property
from collections import defaultdict

import pandas as pd
from dol import cache_this
from lkj import clog
from i2 import Sig
from tabled import DfFiles

from nltk.corpus import wordnet as wn  # pip install nltk

DFLT_ROOTDIR = os.getenv('WORDNET_WORDS')


_WORD_RE = re.compile(r"[A-Za-z]+")


# --------------------------------------------------------------------------------------
# GlossTag utilities
#
# GlossTag provides sense-tagged versions of WordNet glosses, where each content word
# has been annotated with its correct WordNet sense key. This enables building a clean
# synset→synset definitional reference graph instead of a lemma→lemma graph.


def synset_to_glosstag(synset) -> str:
    """
    Convert a Synset (or synset name like 'entity.n.01') to a GlossTag id like 'n00001740'.

    Examples
    --------
    >>> synset_to_glosstag('dog.n.01')
    'n02084071'
    """
    if isinstance(synset, str):
        synset = wn.synset(synset)

    pos = synset.pos()
    if pos == "s":  # adjective satellite → treat as adjective
        pos = "a"

    offset = synset.offset()
    return f"{pos}{offset:08d}"


def glosstag_to_synset(glosstag: str) -> str:
    """
    Convert GlossTag id like 'n00001740' to a synset name like 'entity.n.01'.

    Examples
    --------
    >>> glosstag_to_synset('n02084071')
    'dog.n.01'
    """
    pos = glosstag[0]
    offset = int(glosstag[1:])
    return wn.synset_from_pos_and_offset(pos, offset).name()


def sense_key_to_synset(sense_key: str) -> str:
    """
    Resolve a WordNet sense key to a synset name.

    Examples
    --------
    >>> sense_key_to_synset("dog%1:05:00::")
    'dog.n.01'
    """
    lemma = wn.lemma_from_key(sense_key)
    return lemma.synset().name()


def _iter_synset_gloss_sense_keys(xml_file, *, include_examples: bool = False):
    """
    Stream-parse a GlossTag merged XML file and yield (source_glosstag_id, sense_keys).

    Parameters
    ----------
    xml_file : file-like object
        An open file or file-like object for the merged XML (e.g., noun.xml).
    include_examples : bool, default False
        If True, extract sense keys from both <def> and <ex> sections.
        If False, extract only from <def>.

    Yields
    ------
    tuple[str, list[str]]
        (source_id, sense_keys) where source_id is a GlossTag id like "n00003553"
        and sense_keys is a list of WordNet sense keys found in the WSD gloss.
    """
    import xml.etree.ElementTree as ET

    context = ET.iterparse(xml_file, events=("start", "end"))
    _, root = next(context)

    current_source = None
    in_wsd_gloss = False
    capture_section = False
    sense_keys = []

    for event, elem in context:
        tag = elem.tag

        if event == "start":
            if tag == "synset":
                current_source = elem.attrib.get("id")
                sense_keys = []
                in_wsd_gloss = False
                capture_section = False

            elif tag == "gloss":
                if elem.attrib.get("desc") == "wsd":
                    in_wsd_gloss = True

            elif in_wsd_gloss and tag in {"def", "ex"}:
                if tag == "def":
                    capture_section = True
                elif tag == "ex":
                    capture_section = bool(include_examples)

            elif capture_section and tag == "id":
                sk = elem.attrib.get("sk")
                if sk:
                    sense_keys.append(sk)

        elif event == "end":
            if tag in {"def", "ex"} and in_wsd_gloss:
                capture_section = False

            elif tag == "gloss" and in_wsd_gloss:
                in_wsd_gloss = False

            elif tag == "synset":
                if current_source is not None:
                    yield current_source, sense_keys

                elem.clear()
                root.clear()


def _words_mentioned_in_text(
    text: str,
    words_set: set[str],
    *,
    max_ngram: int = 1,
    match_policy: str = 'longest_per_span',
) -> set[str]:
    """Extract words (and optional underscore-joined ngrams) mentioned in text.

    >>> ws = {'new', 'york', 'new_york'}
    >>> _words_mentioned_in_text('New York is big', ws, max_ngram=2, match_policy='longest_per_span')
    {'new_york'}
    >>> _words_mentioned_in_text('New York is big', ws, max_ngram=2, match_policy='all') == {'new', 'york', 'new_york'}
    True
    """
    if not text:
        return set()

    tokens = [t.casefold() for t in _WORD_RE.findall(text.replace('-', ' '))]
    if not tokens:
        return set()

    max_ngram = max(1, int(max_ngram))
    candidates: list[tuple[int, int, str]] = []
    for i in range(len(tokens)):
        for n in range(1, min(max_ngram, len(tokens) - i) + 1):
            w = '_'.join(tokens[i : i + n]) if n > 1 else tokens[i]
            if w in words_set:
                candidates.append((i, i + n, w))

    if match_policy == 'all':
        return {w for _, _, w in candidates}
    if match_policy != 'longest_per_span':
        raise ValueError(f"Unknown {match_policy=}. Use 'longest_per_span' or 'all'.")

    # Greedy longest-non-overlapping selection.
    candidates.sort(key=lambda t: (-(t[1] - t[0]), t[0], t[1], t[2]))
    chosen: list[tuple[int, int, str]] = []
    for a, b, w in candidates:
        if all(b <= aa or a >= bb for aa, bb, _ in chosen):
            chosen.append((a, b, w))
    return {w for _, _, w in chosen}


class WordsDacc:
    word_frequency_data_url = 'https://github.com/thorwhalen/content/raw/refs/heads/master/tables/csv/zip/english-word-frequency.csv.zip'
    glosstag_data_url = (
        'https://wordnetcode.princeton.edu/glosstag-files/WordNet-3.0-glosstag.zip'
    )

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

    @cache_this(cache='df_files', key='_word_counts.parquet')
    def _word_counts(self):
        return pd.read_csv(
            self.word_frequency_data_url, keep_default_na=False, na_values=[]
        )

    @cache_this  # keep in RAM
    def word_counts(self):
        # Note: The (..., keep_default_na=False, na_values=[]) is to avoid words "null" and "nan" being interpretted as NaN
        #    see https://www.skytowner.com/explore/preventing_strings_from_getting_parsed_as_nan_for_read_csv_in_pandas
        return self._word_counts.set_index('word')['count']

    @cache_this(cache='df_files', key='_wordnet_words.parquet')
    def _wordnet_words(self):
        return sorted(list(wn.all_lemma_names()))

    @cache_this
    def wordnet_words(self):
        return self._wordnet_words.values[:, 0]

    @property
    def glosstag_zip_file(self):
        from graze import graze

        return graze(self.glosstag_data_url, return_filepaths=True)

    @cache_this(cache='df_files', key='glosstag_edges.parquet')
    def glosstag_edges(
        self,
        *,
        include_examples: bool = False,
        dedup_edges: bool = True,
    ):
        """
        Build synset→synset edges from GlossTag sense-tagged glosses.

        This uses the Princeton GlossTag corpus which provides word sense disambiguation
        for WordNet glosses. Each content word in a gloss has been annotated with its
        correct WordNet sense key, allowing us to build a clean synset→synset
        definitional reference graph.

        Parameters
        ----------
        include_examples : bool, default False
            If True, also include sense keys from <ex> (example) sections.
        dedup_edges : bool, default True
            If True, drop duplicate (source, target) rows.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['source', 'target'] containing synset names.
        """
        import zipfile
        import xml.etree.ElementTree as ET

        self.log('computing glosstag_edges...')

        zip_path = self.glosstag_zip_file
        merged_path_prefix = 'WordNet-3.0/glosstag/merged/'
        filenames = ['noun.xml', 'verb.xml', 'adj.xml', 'adv.xml']

        rows = []

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for filename in filenames:
                xml_path = merged_path_prefix + filename
                self.log(f'  processing {filename}...')

                with zf.open(xml_path) as xml_file:
                    for source_glosstag, sense_keys in _iter_synset_gloss_sense_keys(
                        xml_file, include_examples=include_examples
                    ):
                        for sk in sense_keys:
                            try:
                                target_synset = sense_key_to_synset(sk)
                                source_synset = glosstag_to_synset(source_glosstag)
                            except Exception:
                                # If a sense key cannot be resolved, skip it
                                continue
                            rows.append(
                                {'source': source_synset, 'target': target_synset}
                            )

        df = pd.DataFrame(rows)

        if dedup_edges and len(df) > 0:
            df = df.drop_duplicates(['source', 'target']).reset_index(drop=True)

        return df

    @cache_this(cache='df_files', key='word_and_synset.parquet')
    def word_and_synset(self):
        """
        Map words to their synsets.

        Returns a DataFrame with columns ['word', 'synset'] showing which synsets
        each word participates in. Multiple rows per word (one per synset).
        """

        def gen():
            for word in self.word_list:
                for synset in wn.synsets(word):
                    yield {
                        'word': word,
                        'synset': synset.name(),
                    }

        df = pd.DataFrame(gen())
        return df

    @cache_this(cache='df_files', key='wordnet_metadata.parquet')
    def wordnet_metadata(self):
        """
        Synset-indexed metadata with lemmas as a column.

        Returns a DataFrame indexed by synset name, with columns including:
        - lemmas: list of lemma strings (words) associated with this synset
        - definition, pos, lexname, etc.: synset attributes
        - definition_mentions: list of synsets mentioned in the definition
        """
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

        # Add lemmas column: list of words (lemmas) for each synset
        synset_to_lemmas = defaultdict(list)
        for word, synset in word_and_synset_df[['word', 'synset']].itertuples(
            index=False
        ):
            if word not in synset_to_lemmas[synset]:
                synset_to_lemmas[synset].append(word)

        synsets_metadata['lemmas'] = synsets_metadata['synset'].map(
            lambda s: synset_to_lemmas.get(s, [])
        )

        # Derived adjacency: link to synsets whose words are mentioned in the definition.
        word_to_synsets = defaultdict(set)
        for word, synset in (
            word_and_synset_df[['word', 'synset']]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            word_to_synsets[word].add(synset)

        def _definition_mentions(row):
            mentioned = _words_mentioned_in_text(
                row.get('definition', '') or '',
                self.words_set,
                max_ngram=2,
                match_policy='longest_per_span',
            )
            targets = set()
            for w in mentioned:
                targets.update(word_to_synsets.get(w, ()))
            targets.discard(row.get('synset', None))
            return sorted(targets)

        synsets_metadata['definition_mentions'] = synsets_metadata.apply(
            _definition_mentions, axis=1
        )

        # Set synset as index
        synsets_metadata = synsets_metadata.set_index('synset')
        assert not any(
            synsets_metadata.index.duplicated()
        ), "There were duplicate synsets in the index"

        return synsets_metadata

    @cache_this(
        cache='df_files',
        key=lambda self, *, max_ngram=2, match_policy='longest_per_span', include_self_loops=False: (
            f"synset_definition_links.ng{int(max_ngram)}.{match_policy}.self{int(include_self_loops)}.parquet"
        ),
    )
    def synset_definition_links(
        self,
        *,
        max_ngram: int = 2,
        match_policy: str = 'longest_per_span',
        include_self_loops: bool = False,
    ):
        """
        Synset-to-synset edges when source definition mentions words that map to target synsets.

        Returns a DataFrame with columns ['source', 'target'] where both are synset names.
        This creates a clean concept graph without exploding edges across all lemma combinations.

        Parameters
        ----------
        max_ngram : int, default 2
            Maximum n-gram length to extract from definitions
        match_policy : str, default 'longest_per_span'
            How to handle overlapping matches ('longest_per_span' or 'all')
        include_self_loops : bool, default False
            Whether to include edges where source == target

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['source', 'target'] containing synset names
        """

        word_to_synsets = defaultdict(set)
        for word, synset in (
            self.word_and_synset[['word', 'synset']]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            word_to_synsets[word].add(synset)

        synset_def = self.wordnet_metadata['definition']

        seen = set()
        rows = []
        for source_synset in synset_def.index:
            definition = synset_def.get(source_synset, '')
            mentioned = _words_mentioned_in_text(
                definition,
                self.words_set,
                max_ngram=max_ngram,
                match_policy=match_policy,
            )

            for mentioned_word in mentioned:
                for target_synset in word_to_synsets.get(mentioned_word, ()):
                    if not include_self_loops and target_synset == source_synset:
                        continue
                    k = (source_synset, target_synset)
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append(
                        {
                            'source': source_synset,
                            'target': target_synset,
                        }
                    )

        return pd.DataFrame(rows)

    @cache_this(cache='df_files', key='words_used_in_definition_of_words.parquet')
    def words_used_in_definition_of_words(self):
        """
        Create (source, target) pairs where source words link to target words used in their definitions.

        This is different from synset_definition_links as it operates purely on words (not synsets/lemmas).
        Each word has a single definition (chosen by smallest lexicographic synset name), and we
        create edges from each word to all words mentioned in its definition (filtered to words
        that have their own definitions).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['source', 'target'] where:
            - source: the word being defined
            - target: a word mentioned in the source word's definition
        """
        # Get unique word metadata
        unique_words_df = self.word_indexed_metadata

        # Get the set of valid words (words that have definitions)
        valid_words = set(unique_words_df.index)

        def tokenize_definition(definition):
            """Extract words from definition, keeping only alphanumeric tokens"""
            tokens = re.findall(r'\b[a-z]+\b', str(definition).lower())
            return tokens

        # Build list of (source, target) pairs
        pairs = []

        for word, row in unique_words_df.iterrows():
            source_word = word
            definition = row['definition']

            # Tokenize and filter
            tokens = tokenize_definition(definition)
            filtered_tokens = [
                t for t in tokens if t in valid_words and t != source_word
            ]

            # Create pairs
            for target_word in filtered_tokens:
                pairs.append({'source': source_word, 'target': target_word})

        return pd.DataFrame(pairs)

    @property
    def word_frequencies(self):
        t = self.word_counts
        t = t / t.sum()
        t.name = "frequency"
        return t

    @cache_this(cache='df_files', key='wordnet_feature_meta.parquet')
    def wordnet_feature_meta(self):
        """
        Synset-indexed feature metadata.

        Returns a DataFrame indexed by synset with feature columns including:
        - definition, pos, lexname, etc. (synset attributes)
        - lemmas: list of words associated with this synset

        Note: This is now synset-indexed. For word-indexed metadata, use word_indexed_metadata.
        """
        # Start with synset-indexed metadata
        df = self.wordnet_metadata.copy()

        # Select only the feature attributes (not collection attributes)
        feature_cols = [col for col in wordnet_feature_attr_names if col in df.columns]
        feature_cols.append('lemmas')  # Include lemmas column

        return df[feature_cols]

    @cache_this(cache='df_files', key='word_indexed_metadata.parquet')
    def word_indexed_metadata(self):
        """
        Word-indexed metadata with unique words (using smallest lexicographic synset name).

        For each word, selects the synset with the smallest lexicographic name and returns
        a DataFrame indexed by word with columns including:
        - synset: the chosen synset name
        - definition, pos, lexname, etc.: synset attributes
        - frequency: word frequency
        - umap_x, umap_y: embedding coordinates

        This provides a single representative synset per word, useful for word-level analysis.
        """
        # Get word-to-synsets mapping
        word_synset_df = self.word_and_synset.copy()

        # For each word, choose the synset with smallest lexicographic name
        word_synset_df = word_synset_df.sort_values(['word', 'synset'])
        chosen_synsets = word_synset_df.groupby('word').first()['synset']

        # Get metadata for chosen synsets
        metadata = self.wordnet_metadata.copy()

        # Create word-indexed dataframe
        result = pd.DataFrame(index=chosen_synsets.index)
        result['synset'] = chosen_synsets

        # Add synset metadata
        for col in metadata.columns:
            if col != 'lemmas':  # Don't need lemmas list in word view
                result[col] = result['synset'].map(metadata[col])

        # Add word frequency
        result['frequency'] = self.word_frequencies

        # Add embeddings
        result = result.merge(
            self.umap_embeddings, left_index=True, right_index=True, how='left'
        )

        return result

    @cache_this(cache='df_files', key='wordnet_collection_meta.parquet')
    def wordnet_collection_meta(self):
        """
        Synset-indexed collection metadata (relationships like hypernyms, hyponyms, etc.).

        Returns a DataFrame indexed by synset name with columns containing:
        - Lists of related synset names (hypernyms, hyponyms, etc.)
        - frequency: sum of word frequencies for all lemmas in the synset
        """
        df = self.wordnet_metadata[wordnet_collection_attr_names].copy()

        # Add frequency column: sum of frequencies for all lemmas in each synset
        freq_dict = self.word_frequencies.to_dict()
        lemmas_series = self.wordnet_metadata['lemmas']

        def _synset_frequency(synset):
            lemmas = lemmas_series.get(synset, [])
            return sum(freq_dict.get(lemma, 0) for lemma in lemmas)

        df['frequency'] = pd.Series(df.index).apply(_synset_frequency).values

        return df

    def compute_and_save_all_link_data(self):
        """
        Compute and save link data for all wordnet relationships of wordnet_collection_meta.

        :param self: Description
        """
        for relationship_name in self.wordnet_collection_meta.columns:
            adjacencies = self.wordnet_collection_meta[relationship_name]

            link_data = pd.DataFrame(
                [
                    {'source': src, 'target': tgt}
                    for src, targets in adjacencies.items()
                    for tgt in targets
                ]
            )
            link_data = link_data.sort_values(['source', 'target']).reset_index(
                drop=True
            )
            self.df_files[f"link_data/{relationship_name}.parquet"] = link_data

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
    def synset_to_lemmas(self):
        """
        Get mapping from synset names to lists of lemmas (words).

        Returns
        -------
        dict
            Dictionary mapping synset name (str) to list of lemma strings
        """
        return self.wordnet_metadata['lemmas'].to_dict()

    @cache_this
    def word_to_synsets(self):
        """
        Get mapping from words to lists of synset names.

        Returns
        -------
        dict
            Dictionary mapping word (str) to set of synset names
        """
        word_to_synsets = defaultdict(set)
        for word, synset in self.word_and_synset[['word', 'synset']].itertuples(
            index=False
        ):
            word_to_synsets[word].add(synset)
        return dict(word_to_synsets)

    @cache_this(cache='df_files', key='synsets_and_lemmas_links.parquet')
    def synsets_and_lemmas_links(self):
        """
        Bipartite graph linking synsets to their lemmas (words).

        Returns a DataFrame with columns ['source', 'target'] where:
        - source: synset name (e.g., "dog.n.01")
        - target: lemma/word (e.g., "dog")

        This represents the bipartite relationship between concepts (synsets)
        and their surface forms (lemmas/words).
        """
        rows = []
        for synset, lemmas in self.synset_to_lemmas.items():
            for lemma in lemmas:
                rows.append({'source': synset, 'target': lemma})
        return pd.DataFrame(rows)

    @cache_this(cache='df_files', key='synsets_and_lemmas_meta.parquet')
    def synsets_and_lemmas_meta(self):
        """
        Metadata for the bipartite synset-lemma graph nodes.

        Returns a DataFrame with columns:
        - item: the synset name or lemma/word
        - kind: "synset" or "lemma" indicating the node type

        This provides node metadata for the bipartite graph returned by
        synsets_and_lemmas_links.
        """
        synsets = list(self.wordnet_metadata.index)
        lemmas = list(self.words_set)

        rows = []
        for synset in synsets:
            rows.append({'item': synset, 'kind': 'synset'})
        for lemma in lemmas:
            rows.append({'item': lemma, 'kind': 'lemma'})

        return pd.DataFrame(rows)

    # DEPRECATED: Use word_indexed_metadata instead
    # This method is kept for backward compatibility but may be removed in future versions
    def words_and_features(self):
        """
        DEPRECATED: Use word_indexed_metadata instead.

        This method provided word-level features but has been superseded by
        word_indexed_metadata which provides a cleaner word-indexed view.
        """
        return self.word_indexed_metadata

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
    Generate metadata dictionaries for WordNet synsets.

    Args:
        synsets: Iterable of synset objects or synset name strings

    Yields:
        dict: A dictionary with synset metadata including:
            - synset: synset name (e.g., "dog.n.01")
            - definition, pos, lexname, etc.: synset attributes
            - lemmas: list of lemma strings (words) in this synset
            - hypernyms, hyponyms, etc.: list of related synset names
    """
    from nltk.corpus import wordnet as wn

    for synset in synsets:
        if isinstance(synset, str):
            try:
                synset = wn.synset(synset)
            except Exception as e:
                raise type(e)(
                    f"Failed to resolve synset name {synset!r}. "
                    "This often happens when cached synset ids were produced with a different WordNet version. "
                    "Try deleting the old cached parquet files (e.g. word_and_synset.parquet / wordnet_metadata.parquet) "
                    "or let the new versioned cache keys recompute fresh. "
                    f"Original error: {e}"
                ) from e

        d = {
            "synset": synset.name(),  # Synset identifier (e.g., "salt.n.03")
            "example": next(iter(synset.examples()), ''),
            "lemmas": [
                lemma.name() for lemma in synset.lemmas()
            ],  # List of word strings
        }
        for attr_name in wordnet_feature_attr_names:
            d[attr_name] = getattr(synset, attr_name)()
        for attr_name in wordnet_collection_attr_names:
            if attr_name == 'definition_mentions':
                continue
            val = getattr(synset, attr_name)()
            try:
                d[attr_name] = [synset.name() for synset in val]
            except Exception as e:
                raise print(f"{attr_name} failed. {val=}. {e=}")

        yield d


import pandas as pd


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

# Derived relationship columns (not native Synset methods)
wordnet_collection_attr_names = sorted(
    set(wordnet_collection_attr_names) | {'definition_mentions'}
)
wordnet_feature_attr_names = sorted(
    set(wordnet_attr_names) - set(wordnet_collection_attr_names)
)


# --------------------------------------------------------------------------------------
# Misc utils
# TODO: Perhaps move to some general graph/network utils (e.g. move to linked)


def thin_links_by_frequency(
    links,
    node_frequency,
    *,
    max_frequency=None,
    max_links=None,
    method='max',  # 'max' or 'sum'
    verbose=False,
):
    """
    Thin out link data by filtering based on node frequency.

    Parameters
    ----------
    links : pd.DataFrame
        DataFrame with 'source' and 'target' columns containing node IDs
    node_frequency : pd.DataFrame or pd.Series
        DataFrame/Series with node IDs as index and 'frequency' column/values
    max_frequency : float, optional
        If provided, filters out links where either source or target has frequency > max_frequency
    max_links : int, optional
        If provided, uses binary search to find the max_frequency threshold that gives
        approximately this many links (as large as possible under this constraint)
    method : str, default 'max'
        How to combine source and target frequencies:
        - 'max': Remove link if max(source_freq, target_freq) > threshold
        - 'sum': Remove link if source_freq + target_freq > threshold
    verbose : bool, default True
        Print progress information

    Returns
    -------
    pd.DataFrame
        Filtered links dataframe
    """
    # Convert to Series if needed
    if isinstance(node_frequency, pd.DataFrame):
        node_freq_series = node_frequency['frequency']
    else:
        node_freq_series = node_frequency

    # Create lookup dictionary for faster access
    freq_dict = node_freq_series.to_dict()

    def filter_links_by_threshold(threshold):
        """Helper to filter links given a threshold."""
        # Map frequencies to source and target
        source_freq = links['source'].map(freq_dict).fillna(0)
        target_freq = links['target'].map(freq_dict).fillna(0)

        # Apply filtering based on method
        if method == 'max':
            mask = (source_freq <= threshold) & (target_freq <= threshold)
        elif method == 'sum':
            mask = (source_freq + target_freq) <= threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        return links[mask]

    # If max_frequency is directly provided
    if max_frequency is not None:
        if verbose:
            print(f"Filtering with max_frequency={max_frequency:.6f}")
        filtered = filter_links_by_threshold(max_frequency)
        if verbose:
            print(
                f"Result: {len(links)} -> {len(filtered)} links ({100*len(filtered)/len(links):.2f}%)"
            )
        return filtered

    # If max_links is provided, use binary search
    if max_links is not None:
        if verbose:
            print(f"Using binary search to find threshold for ~{max_links:,} links...")

        # Get frequency range for binary search
        min_freq = node_freq_series.min()
        max_freq = node_freq_series.max()

        if method == 'sum':
            # For sum method, max threshold is 2x max_freq
            max_freq = max_freq * 2

        # Binary search
        left, right = min_freq, max_freq
        best_threshold = max_freq
        best_filtered = links

        iterations = 0
        max_iterations = 50

        while iterations < max_iterations and right - left > 1e-8:
            mid = (left + right) / 2
            filtered = filter_links_by_threshold(mid)
            n_links = len(filtered)

            if verbose and iterations % 5 == 0:
                print(
                    f"  Iteration {iterations}: threshold={mid:.6f}, n_links={n_links:,}"
                )

            if n_links <= max_links:
                # We're under the limit, this is a candidate
                if n_links > len(best_filtered) or len(best_filtered) > max_links:
                    best_threshold = mid
                    best_filtered = filtered
                # Try to increase threshold to get more links
                left = mid
            else:
                # Too many links, decrease threshold
                right = mid

            iterations += 1

        if verbose:
            print(f"\nFinal result: threshold={best_threshold:.6f}")
            print(
                f"Links: {len(links):,} -> {len(best_filtered):,} ({100*len(best_filtered)/len(links):.2f}%)"
            )

        return best_filtered

    raise ValueError("Must provide either max_frequency or max_links")
