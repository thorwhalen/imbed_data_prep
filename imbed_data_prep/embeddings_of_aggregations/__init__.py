"""Tools to analyze the embeddings of aggregations"""

import numpy as np
from dol import Pipe
import pandas as pd
from typing import List, TypeVar, Tuple
from collections.abc import Mapping, Callable, Iterable
import oa
from imbed.base import simple_semantic_features
from imbed.util import fuzzy_induced_graph as fuzzy_induced_graph, Node, Nodes

# DFLT_EMBEDDING_FUNC = oa.embeddings
DFLT_EMBEDDING_FUNC = simple_semantic_features
DFLT_RANDOM_SEED = 0


def get_n_unique_permutations(arr, n: int, seed: int = DFLT_RANDOM_SEED):
    """
    Get n unique permutations of an array, with a random seed fixed, and
    raise an error if n is larger than the number of possible permutations.

    Args:
        arr (list): The list to permute.
        n (int): The number of unique permutations to generate.
        seed (int): The random seed for reproducibility.

    Returns:
        list: A list of unique permutations.

    Raises:
        ValueError: If n is larger than the number of possible permutations.

    Examples:

        >>> get_n_unique_permutations([1, 2, 3], 2)
        [(3, 2, 1), (3, 1, 2)]
        >>> get_n_unique_permutations([1, 2, 3], 2, seed=0)  # default is seed=0
        [(3, 2, 1), (3, 1, 2)]
        >>> get_n_unique_permutations([1, 2, 3], 2, seed=1)  # but if you change seed...
        [(2, 3, 1), (1, 3, 2)]
        >>> get_n_unique_permutations([1, 2, 3], 6)
        [(1, 3, 2), (1, 2, 3), (2, 1, 3), (3, 2, 1), (3, 1, 2), (2, 3, 1)]
        >>> get_n_unique_permutations([1, 2, 3], 7)
        Traceback (most recent call last):
        ...
        ValueError: n (=7) is larger than the number of possible permutations: 6
    """
    import numpy as np
    import math

    np.random.seed(seed)

    n_perms = math.factorial(len(arr))
    if n > n_perms:
        raise ValueError(
            f"n (={n}) is larger than the number of possible permutations: {n_perms}"
        )
    perms = set()
    while len(perms) < n:
        perms.add(tuple(np.random.permutation(arr)))
    return list(perms)


def aggregated_embeddings_for_sample(
    graph: Mapping[Node, Nodes],
    n_nodes: int,
    n_permutations: int,
    *,
    node_to_text: Callable[[Node], str],
    aggregate_texts: callable = "\n\n".join,
    text_to_embedding: callable = None,
    max_permutations: int = 100,
    seed: int = 0,
):
    """
    Get a (without replacement) sample of n_nodes items of the citation_graph
    and for each, take n_permutations permutations of the cited_ids,
    aggregate the titles of the cited_ids and compute its embedding.

    Args:
        graph (dict): The citation graph.
        n_nodes (int): The number of nodes to sample.
        n_permutations (int): The number of permutations to take for each cited_ids list.
        node_to_text (callable): A function that takes a node (ID) and returns the text to embed.
        aggregate_texts (callable): A function that takes a list of texts and returns a single aggregated text.
        text_to_embedding (callable): A function that takes a text and returns an embedding.
        max_permutations (int): The maximum number of permutations to take.
        seed (int): The random seed for reproducibility.

    Returns:
        An iterable of dicts with aggregated embeddings for each sample.


    Examples:

        >>> graph = {
        ...     'paper1': ['paper2', 'paper3'],
        ...     'paper2': ['paper3'],
        ...     'paper3': ['paper1'],
        ...     'paper4': [],
        ... }
        >>> node_titles = {
        ...     'paper1': 'Title of Paper 1',
        ...     'paper2': 'Title of Paper 2',
        ...     'paper3': 'Title of Paper 3',
        ...     'paper4': 'Title of Paper 4',
        ... }
        >>> n_nodes = 2
        >>> n_permutations = 2
        >>> list(aggregated_embeddings_for_sample(
        ...     graph, n_nodes, n_permutations, node_to_text=node_titles.get,
        ...     text_to_embedding=simple_semantic_features, seed=42
        ... ))  # doctest: +SKIP
        [{'citing_id': 'paper2',
        'permutation_index': 0,
        'aggregated_title': 'Title of Paper 2\n\nTitle of Paper 3',
        'embedding': (8, 26, 0)},
        {'citing_id': 'paper1',
        'permutation_index': 0,
        'aggregated_title': 'Title of Paper 1\n\nTitle of Paper 2\n\nTitle of Paper 3',
        'embedding': (12, 39, 0)},
        {'citing_id': 'paper1',
        'permutation_index': 1,
        'aggregated_title': 'Title of Paper 1\n\nTitle of Paper 3\n\nTitle of Paper 2',
        'embedding': (12, 39, 0)}]

    """
    import random
    from math import factorial

    nodes = list(graph.keys())
    if n_nodes > len(nodes):
        raise ValueError(
            f"n_nodes ({n_nodes}) is larger than the number of nodes in the citation_graph ({len(nodes)})"
        )

    np.random.seed(seed)
    sampled_nodes = random.sample(nodes, n_nodes)

    for citing_id in sampled_nodes:
        neighbor_nodes = graph[citing_id]
        if len(neighbor_nodes) == 0:
            continue  # skip nodes with no citations

        citing_title = node_to_text(citing_id)

        n_perms = min(
            min(n_permutations, max_permutations), factorial(len(neighbor_nodes))
        )
        perms = get_n_unique_permutations(neighbor_nodes, n_perms, seed=seed)

        for idx, perm in enumerate(perms):
            aggregated_title = aggregate_texts(
                [citing_title] + [node_to_text(neighbor_node) for neighbor_node in perm]
            )

            d = {
                'citing_id': citing_id,
                'n_cited': len(neighbor_nodes),
                'permutation_index': idx,
                'aggregated_title': aggregated_title,
            }
            if text_to_embedding:
                d['embedding'] = text_to_embedding(aggregated_title)
            yield d


get_aggregated_embeddings_for_sample = Pipe(
    aggregated_embeddings_for_sample, list, pd.DataFrame
)

# -------------------------------------------------------------------------------------
# Tests

from imbed.tests.utils_for_tests import simple_semantic_features


def test_get_n_unique_permutations():
    arr = [1, 2, 3]
    n = 2
    perms = get_n_unique_permutations(arr, n, seed=0)
    expected_perms = [(3, 2, 1), (3, 1, 2)]
    assert sorted(perms) == sorted(
        expected_perms
    ), "Permutations do not match expected output"


def test_get_n_unique_permutations_error():
    arr = [1, 2, 3]
    n = 7  # There are only 6 possible permutations
    try:
        perms = get_n_unique_permutations(arr, n)
    except ValueError as e:
        assert str(e) == "n (=7) is larger than the number of possible permutations: 6"
    else:
        assert False, "ValueError was not raised when expected"


from collections.abc import Sequence


def _is_vector(v):
    if not isinstance(v, Sequence):
        return False
    else:
        first_element = next(iter(v), None)
        return isinstance(first_element, (int, float))


def test_get_aggregated_embeddings_for_sample():
    citation_graph = {
        'paper1': ['paper2', 'paper3'],
        'paper2': ['paper3'],
        'paper3': ['paper1'],
        'paper4': [],
    }

    node_titles = {
        'paper1': 'Title of Paper 1',
        'paper2': 'Title of Paper 2',
        'paper3': 'Title of Paper 3',
        'paper4': 'Title of Paper 4',
    }

    n_nodes = 2
    n_permutations = 2

    df = get_aggregated_embeddings_for_sample(
        citation_graph,
        n_nodes,
        n_permutations,
        node_to_text=node_titles.get,
        text_to_embedding=simple_semantic_features,
    )

    assert not df.empty, "DataFrame is empty"
    assert len(df) <= n_nodes * n_permutations, "DataFrame has more rows than expected"

    # Check that embeddings are numpy arrays
    import numpy as np

    assert all(map(_is_vector, df['embedding'])), "Embeddings are not vectors"


def test_node_with_no_citations():
    citation_graph = {
        'paper1': ['paper2', 'paper3'],
        'paper2': ['paper3'],
        'paper3': ['paper1'],
        'paper4': [],
    }

    node_titles = {
        'paper1': 'Title of Paper 1',
        'paper2': 'Title of Paper 2',
        'paper3': 'Title of Paper 3',
        'paper4': 'Title of Paper 4',
    }

    n_nodes = 4  # All nodes
    n_permutations = 2

    df = get_aggregated_embeddings_for_sample(
        citation_graph,
        n_nodes,
        n_permutations,
        node_to_text=node_titles.get,
        text_to_embedding=simple_semantic_features,
    )

    # Check that 'paper4' (node with no citations) is not in df['node_id']
    assert (
        'paper4' not in df['node_id'].values
    ), "Node with no citations should be skipped"


def test_n_nodes_too_large():
    citation_graph = {
        'paper1': ['paper2', 'paper3'],
        'paper2': ['paper3'],
        'paper3': ['paper1'],
        'paper4': [],
    }

    node_titles = {
        'paper1': 'Title of Paper 1',
        'paper2': 'Title of Paper 2',
        'paper3': 'Title of Paper 3',
        'paper4': 'Title of Paper 4',
    }

    n_nodes = 5  # There are only 4 nodes in citation_graph
    n_permutations = 2

    try:
        df = get_aggregated_embeddings_for_sample(
            citation_graph,
            n_nodes,
            n_permutations,
            node_to_text=node_titles.get,
            text_to_embedding=simple_semantic_features,
        )
    except ValueError as e:
        assert (
            "n_nodes (5) is larger than the number of nodes in the citation_graph (4)"
            in str(e)
        )
    else:
        assert False, "ValueError was not raised when expected"


def test_all_permutation_tools():
    test_get_n_unique_permutations()
    test_get_n_unique_permutations_error()
    test_get_aggregated_embeddings_for_sample()
    test_node_with_no_citations()
    test_n_nodes_too_large()
