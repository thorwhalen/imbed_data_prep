# WordNet Words Refactor Summary

## Overview

The `wordnet_words.py` module has been refactored to use **synsets as the primary index** instead of lemma-in-synset "sense strings" (e.g., `angstrom.n.01.a`). This design avoids edge explosion and creates a cleaner semantic graph architecture.

## Key Architecture Changes

### Before (Lemma-Based)
- Primary index: "sense strings" like `angstrom.n.01.a` (lemma-in-synset)
- Replicated synset data for each lemma
- Link edges: (lemma, lemma) pairs with synset annotations
- Edge count: O(|synsets| × avg_lemmas²) due to Cartesian expansion

### After (Synset-Based)
- Primary index: synset names like `angstrom.n.01`
- Single row per synset with `lemmas` column
- Link edges: (synset, synset) pairs only
- Edge count: O(|synsets|) - clean concept graph
- Bipartite layer: `word_and_synset` provides word↔synset mapping

## Data Structure Changes

### 1. `word_and_synset` (Bipartite Mapping)
**Before:**
```
Index: lemma (sense string)
Columns: ['word', 'synset', 'lemma']
```

**After:**
```
Columns: ['word', 'synset']
No index - just a bipartite mapping table
```

### 2. `wordnet_metadata` (Core Metadata)
**Before:**
```
Index: lemma (sense string like 'dog.n.01.domestic_dog')
Columns: [word, synset, definition, pos, ...]
Multiple rows per synset (one per lemma)
```

**After:**
```
Index: synset (like 'dog.n.01')
Columns: [lemmas, definition, pos, ...]
  - lemmas: ['dog', 'domestic_dog', 'canis_familiaris']
Single row per synset
```

### 3. `synset_definition_links` (Concept Graph)
**Before:**
```
Columns: ['source_synset', 'target_synset', 'source_word', 'target_word']
Multiple edges per synset pair (one per lemma combination)
Edge count: ~millions (inflated)
```

**After:**
```
Columns: ['source', 'target']
Both are synset names
Single edge per synset pair
Edge count: 3.3M (clean concept graph)
```

### 4. `word_indexed_metadata` (Word View)
**New structure:**
```
Index: word
Columns: ['synset', 'definition', 'pos', 'frequency', 'umap_x', 'umap_y', ...]
One row per word (uses smallest lexicographic synset)
```

### 5. `words_used_in_definition_of_words` (Word→Word Links)
**Unchanged:**
```
Columns: ['source', 'target']
Both are words (not synsets)
Edge count: 350K
```

## Helper Methods

### Before
- `lemmas_of_synset()`: synset → set of sense strings
- `word_of_lemma()`: sense string → word
- `lemma_graph_edges()`: expand synset adjacency to lemma pairs

### After
- `synset_to_lemmas()`: synset → list of words
- `word_to_synsets()`: word → set of synsets
- (Removed `lemma_graph_edges` - use synset graph directly)

## Cache Key Changes

All cache keys have been versioned to `.v2.parquet` to force recomputation:
- `word_and_synset.v2.parquet`
- `wordnet_metadata.v2.parquet`
- `wordnet_feature_meta.v2.parquet`
- `word_indexed_metadata.v2.parquet`
- `synset_definition_links.v2.ng2.longest_per_span.self0.parquet`
- `words_used_in_definition_of_words.v2.parquet`
- `wordnet_collection_meta.v2.parquet`

## Example Data

### word_and_synset (Bipartite)
```
  word                             synset
0    a                      angstrom.n.01
1    a                     vitamin_a.n.01
2    a  deoxyadenosine_monophosphate.n.01
3    a                       adenine.n.01
4    a                        ampere.n.02
```

### wordnet_metadata (Synset-Indexed)
```
Index: angstrom.n.01
lemmas: [a, angstrom, as]
definition: a metric unit of length equal to one ten billionth of a meter...
pos: noun
lexname: noun.quantity
hypernyms: [metric_linear_unit.n.01]
part_holonyms: [nanometer.n.01]
part_meronyms: [picometer.n.01]
...
```

### synset_definition_links (Concept Graph)
```
          source                   target
0  angstrom.n.01              micron.n.01
1  angstrom.n.01           radiation.n.02
2  angstrom.n.01        radiotherapy.n.01
3  angstrom.n.01  radiation_sickness.n.01
4  angstrom.n.01           radiation.n.04
```

### word_indexed_metadata (Word View)
```
Index: a
synset: a.n.06
definition: the 1st letter of the Roman alphabet
pos: noun
frequency: 0.015441
umap_x: 3.027916
umap_y: 3.760965
...
```

## Benefits

1. **Scalability**: Concept graph edges are O(|synsets|) not O(|synsets| × lemmas²)
2. **Clarity**: Semantic structure (synsets) is separate from lexical layer (words)
3. **Efficiency**: No redundant synset data across lemmas
4. **Flexibility**: Can expand to lemmas at render time when needed
5. **Clean APIs**:
   - Work with concepts → use synset-indexed data
   - Work with words → use word-indexed views
   - Map between layers → use bipartite mapping

## Migration Notes

- Old cached `.parquet` files will be ignored (new `.v2.` versions)
- `words_and_features()` now deprecated (use `word_indexed_metadata`)
- Code working with lemma indices needs to switch to synset or word indexing
- Link traversal now cleaner: synset→synset, then optionally expand to lemmas

## Test Results

All tests pass ✓:
- `word_and_synset`: 123,691 word-synset pairs
- `wordnet_metadata`: 74,192 synsets with lemmas
- `synset_definition_links`: 3.3M synset→synset edges
- `word_indexed_metadata`: 52,078 unique words
- `words_used_in_definition_of_words`: 350K word→word edges
- Helper methods return proper dict mappings
