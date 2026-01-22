"""
Explore the refactored WordsDacc data to see the synset-based architecture.
"""

from imbed_data_prep.wordnet_words import WordsDacc
from config2py import config_getter

rootdir = config_getter('WORDNET_WORDS_PROJECT_ROOTDIR')
dacc = WordsDacc(rootdir)

print("="*80)
print("EXAMPLE: The word 'dog' in the refactored architecture")
print("="*80)

# 1. Find all synsets for 'dog'
word_synsets = dacc.word_to_synsets
dog_synsets = word_synsets.get('dog', set())
print(f"\n1. Synsets for 'dog': {len(dog_synsets)} synsets")
for synset in sorted(dog_synsets)[:5]:
    print(f"   - {synset}")

# 2. Look at metadata for one synset
metadata = dacc.wordnet_metadata
if 'dog.n.01' in metadata.index:
    print(f"\n2. Metadata for 'dog.n.01' synset:")
    dog_meta = metadata.loc['dog.n.01']
    print(f"   Lemmas: {dog_meta['lemmas']}")
    print(f"   Definition: {dog_meta['definition'][:80]}...")
    print(f"   POS: {dog_meta['pos']}")
    print(f"   Hypernyms: {dog_meta['hypernyms']}")
    print(f"   Hyponyms (first 5): {dog_meta['hyponyms'][:5]}")

# 3. Concept graph: what synsets does dog.n.01 link to?
links = dacc.synset_definition_links()
dog_links = links[links['source'] == 'dog.n.01']
print(f"\n3. Synset-level definition links from 'dog.n.01': {len(dog_links)} links")
print(f"   Target synsets: {list(dog_links['target'].head(10))}")

# 4. Word-level view
word_meta = dacc.word_indexed_metadata
if 'dog' in word_meta.index:
    print(f"\n4. Word-indexed view for 'dog':")
    dog_word = word_meta.loc['dog']
    print(f"   Chosen synset: {dog_word['synset']}")
    print(f"   Definition: {dog_word['definition'][:80]}...")
    print(f"   Frequency: {dog_word['frequency']:.6f}")
    print(f"   UMAP coords: ({dog_word['umap_x']:.2f}, {dog_word['umap_y']:.2f})")

# 5. Word-to-word definition links
word_links = dacc.words_used_in_definition_of_words
dog_word_links = word_links[word_links['source'] == 'dog']
print(f"\n5. Word-level definition links from 'dog': {len(dog_word_links)} links")
print(f"   Target words: {list(dog_word_links['target'].head(15))}")

print("\n" + "="*80)
print("ARCHITECTURE DEMONSTRATION")
print("="*80)

print("""
The layered architecture:

1. BIPARTITE LAYER (word ↔ synset)
   - word_and_synset: maps words to their synsets
   - word_to_synsets: dict for quick word → synsets lookup
   - synset_to_lemmas: dict for quick synset → words lookup

2. CONCEPT LAYER (synset-indexed)
   - wordnet_metadata: one row per synset, with lemmas column
   - synset_definition_links: clean synset→synset graph
   - Edge count: O(|synsets|) not O(|synsets| × lemmas²)

3. WORD LAYER (word-indexed views)
   - word_indexed_metadata: one representative synset per word
   - words_used_in_definition_of_words: word→word graph

Benefits:
✓ No edge explosion from lemma combinations
✓ Semantic graph (concepts) separated from lexical layer (words)
✓ Can expand to lemmas at render time when needed
✓ Much more scalable and interpretable
""")

print("\n" + "="*80)
print("STATISTICS")
print("="*80)

print(f"Total words: {len(dacc.word_list):,}")
print(f"Total synsets: {len(metadata):,}")
print(f"Avg lemmas per synset: {sum(len(l) for l in metadata['lemmas']) / len(metadata):.2f}")
print(f"Synset→synset edges: {len(links):,}")
print(f"Word→word edges: {len(word_links):,}")
print(f"Edge reduction factor: {len(links) / len(word_links):.1f}x more synset edges")
print(f"  (shows that synset graph captures richer semantic structure)")
