"""Quick test of the refactored WordsDacc code."""

from imbed_data_prep.wordnet_words import WordsDacc
from config2py import config_getter

# Get rootdir
rootdir = config_getter('WORDNET_WORDS_PROJECT_ROOTDIR')

print("Creating WordsDacc instance...")
dacc = WordsDacc(rootdir)

print("\n" + "="*80)
print("TEST 1: word_and_synset (bipartite mapping)")
print("="*80)
word_synset = dacc.word_and_synset
print(f"Shape: {word_synset.shape}")
print(f"Columns: {list(word_synset.columns)}")
print(f"\nFirst 5 rows:")
print(word_synset.head())
print(f"\nExpected columns: ['word', 'synset']")
print(f"✓ PASS" if list(word_synset.columns) == ['word', 'synset'] else "✗ FAIL")

print("\n" + "="*80)
print("TEST 2: wordnet_metadata (synset-indexed)")
print("="*80)
metadata = dacc.wordnet_metadata
print(f"Shape: {metadata.shape}")
print(f"Index dtype: {metadata.index.dtype}")
print(f"'lemmas' in columns: {'lemmas' in metadata.columns}")
print(f"\nFirst row:")
print(metadata.iloc[0])
print(f"\nExpected: synset-indexed with 'lemmas' column")
print(f"✓ PASS" if 'lemmas' in metadata.columns else "✗ FAIL")

print("\n" + "="*80)
print("TEST 3: synset_definition_links (synset→synset)")
print("="*80)
links = dacc.synset_definition_links()  # Call the method with default parameters
print(f"Shape: {links.shape}")
print(f"Columns: {list(links.columns)}")
print(f"\nFirst 5 links:")
print(links.head())
print(f"\nExpected columns: ['source', 'target']")
print(f"✓ PASS" if list(links.columns) == ['source', 'target'] else "✗ FAIL")

# Verify synset format
sample_source = links.iloc[0]['source']
sample_target = links.iloc[0]['target']
print(f"\nSample source: {sample_source}")
print(f"Sample target: {sample_target}")
print(f"Source is synset format: {'.' in sample_source}")
print(f"Target is synset format: {'.' in sample_target}")

print("\n" + "="*80)
print("TEST 4: word_indexed_metadata")
print("="*80)
word_meta = dacc.word_indexed_metadata
print(f"Shape: {word_meta.shape}")
print(f"Index name: {word_meta.index.name}")
print(f"'synset' in columns: {'synset' in word_meta.columns}")
print(f"\nFirst row:")
print(word_meta.iloc[0])
print(f"\nExpected: word-indexed with 'synset' column")
print(f"✓ PASS" if 'synset' in word_meta.columns else "✗ FAIL")

print("\n" + "="*80)
print("TEST 5: words_used_in_definition_of_words")
print("="*80)
word_links = dacc.words_used_in_definition_of_words
print(f"Shape: {word_links.shape}")
print(f"Columns: {list(word_links.columns)}")
print(f"\nFirst 5 links:")
print(word_links.head())
print(f"\nExpected columns: ['source', 'target']")
print(f"✓ PASS" if list(word_links.columns) == ['source', 'target'] else "✗ FAIL")

print("\n" + "="*80)
print("TEST 6: Helper methods")
print("="*80)
synset_lemmas = dacc.synset_to_lemmas
word_synsets = dacc.word_to_synsets
print(f"synset_to_lemmas type: {type(synset_lemmas)}")
print(f"word_to_synsets type: {type(word_synsets)}")
print(f"✓ PASS" if isinstance(synset_lemmas, dict) and isinstance(word_synsets, dict) else "✗ FAIL")

print("\n" + "="*80)
print("SUMMARY: All basic tests completed!")
print("="*80)
