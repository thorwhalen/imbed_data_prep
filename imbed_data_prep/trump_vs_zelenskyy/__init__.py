"""Analysis of the Trump vs. Zelenskyy meeting at the White House, Feb 2025."""

"""
Trump-Zelenskyy transcript data access component.

This module provides a data access component for working with 
Trump-Zelenskyy transcript data, with support for embeddings, 
sentiment analysis, and various projections.
"""

import re
import os
from dataclasses import dataclass, field, KW_ONLY
from typing import Optional, List, Union, Dict, Any, Tuple
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import requests
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from dol import cache_this, Files
from dol import add_extension
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _cache_this(func=None, *, cache=None, key=None):
    """Wrapper around cache_this to handle the case where it might not be available."""
    try:
        from dol import cache_this as _cache
        return _cache(func, cache=cache, key=key)
    except ImportError:
        from functools import lru_cache
        if func is None:
            return lambda f: lru_cache(maxsize=128)(f)
        return lru_cache(maxsize=128)(func)


@dataclass
class TrumpZelenskyyDacc:
    """
    Data access component for Trump-Zelenskyy transcript data.
    
    This class handles fetching, processing, and analyzing the transcript data
    with various features like embeddings, sentiment analysis, and dimensionality
    reduction.
    
    Examples:
        >>> dacc = TrumpZelenskyyDacc()
        >>> transcript_df = dacc.transcript_df
        >>> tsne_df = dacc.tsne_projection
    """
    raw_src_url: str = "https://raw.githubusercontent.com/thorwhalen/content/refs/heads/master/text/trump-zelensky-2025-03-01--with_speakers.txt"
    rootdir: str = "."
    top_n_speakers: int = 3
    _: KW_ONLY
    embeddings_save_key: str = 'data/trump_vs_zelenskyy_embeddings.parquet'
    transcript_save_key: str = 'data/trump_vs_zelenskyy_transcript.parquet'
    tsne_components: int = 2
    tsne_random_state: int = 42
    lda_components: int = 2
    pca_components: int = 2
    pca_for_lda_components: int = 50
    speaker_pattern: str = r'\[(?P<speaker>[^\]]+)\]: (?P<text>.*)'
    other_speaker_label: str = 'Other'
    embeddings_provider: Callable | None = None
    verbose: int = 1
    saves_dir: str | None = None
    
    # Internal fields
    _store: Any = field(init=False, default=None)
    _embeddings_vectors: Any = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize the store and create necessary directories."""
        # Make sure data directory exists
        os.makedirs(os.path.join(self.rootdir, os.path.dirname(self.transcript_save_key)), 
                   exist_ok=True)
        os.makedirs(os.path.join(self.rootdir, os.path.dirname(self.embeddings_save_key)), 
                   exist_ok=True)
        
        # Set up the store
        from tabled import DfFiles
        self._store = DfFiles(self.rootdir)
    
    @property
    def store(self):
        """Get the data store."""
        return self._store
    
    @cache_this(cache='_store', key=add_extension('.txt'))
    def fetch_transcript_text(self):
        """
        Fetch the raw transcript text from the source URL.
        
        Returns:
            str: The raw transcript text.
        """
        if self.verbose > 0:
            print(f"Fetching transcript text from {self.raw_src_url}")
        return requests.get(self.raw_src_url).text
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def parse_transcript(self):
        """
        Parse the transcript text into a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with 'speaker' and 'text' columns.
        """
        if self.verbose > 0:
            print("Parsing transcript text")
        
        transcript_text = self.fetch_transcript_text()
        pattern = re.compile(self.speaker_pattern)
        
        transcript_dict_list = [
            match.groupdict()
            for match in pattern.finditer(transcript_text)
        ]
        
        return pd.DataFrame(transcript_dict_list)
    
    @property
    def raw_transcript_df(self):
        """
        Get the raw transcript DataFrame without any processing.
        
        Returns:
            pd.DataFrame: Raw transcript DataFrame.
        """
        return self.parse_transcript()
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def process_transcript(self):
        """
        Process the transcript by replacing minor speakers with 'Other'.
        
        Returns:
            pd.DataFrame: Processed transcript DataFrame.
        """
        if self.verbose > 0:
            print(f"Processing transcript, keeping top {self.top_n_speakers} speakers")
        
        df = self.parse_transcript()
        
        # Replace all speakers that are not the top N with 'Other'
        top_speakers = df['speaker'].value_counts().head(self.top_n_speakers).index
        df['speaker'] = df['speaker'].apply(
            lambda speaker: speaker if speaker in top_speakers else self.other_speaker_label
        )
        
        # Add turn column
        df['turn'] = range(len(df))
        
        return df
    
    @property
    def transcript_df(self):
        """
        Get the processed transcript DataFrame.
        
        Returns:
            pd.DataFrame: Processed transcript DataFrame.
        """
        # Check if the transcript is already saved
        if self.transcript_save_key in self.store:
            return self.store[self.transcript_save_key]
        
        return self.process_transcript()
    
    def compute_embeddings(self):
        """
        Compute embeddings for the transcript text.
        
        Returns:
            np.ndarray: Array of embedding vectors.
        """
        if self.verbose > 0:
            print("Computing embeddings")
        
        if self.embeddings_provider is None:
            try:
                import oa
                embeddings_func = oa.embeddings
            except ImportError:
                raise ImportError("No embeddings provider specified and 'oa' package not available. "
                                 "Please specify an embeddings_provider function.")
        else:
            embeddings_func = self.embeddings_provider
        
        df = self.process_transcript()
        return embeddings_func(df['text'])
    
    @property
    def embeddings(self):
        """
        Get the transcript embeddings.
        
        Returns:
            np.ndarray: Array of embedding vectors.
        """
        if self._embeddings_vectors is not None:
            return self._embeddings_vectors
        
        # Check if embeddings are already saved
        if self.embeddings_save_key in self.store:
            embeddings_df = self.store[self.embeddings_save_key]
            self._embeddings_vectors = np.vstack(embeddings_df['embeddings'])
        else:
            # Compute embeddings
            embeddings_vectors = self.compute_embeddings()
            embeddings_df = pd.DataFrame({'embeddings': embeddings_vectors})
            self.store[self.embeddings_save_key] = embeddings_df
            self._embeddings_vectors = np.vstack(embeddings_vectors)
        
        return self._embeddings_vectors
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def compute_tsne_projection(self):
        """
        Compute t-SNE projection of the embeddings.
        
        Returns:
            pd.DataFrame: DataFrame with t-SNE projection columns.
        """
        if self.verbose > 0:
            print(f"Computing t-SNE projection with {self.tsne_components} components")
        
        embeddings = self.embeddings
        tsne = TSNE(n_components=self.tsne_components, random_state=self.tsne_random_state)
        tsne_vectors = tsne.fit_transform(embeddings)
        
        return pd.DataFrame(
            tsne_vectors, 
            columns=[f'tsne_{i+1}' for i in range(self.tsne_components)]
        )
    
    @property
    def tsne_projection(self):
        """
        Get the t-SNE projection DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with t-SNE projection columns.
        """
        return self.compute_tsne_projection()
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def compute_lda_projection(self):
        """
        Compute LDA projection of the embeddings based on speakers.
        
        Returns:
            pd.DataFrame: DataFrame with LDA projection columns.
        """
        if self.verbose > 0:
            print(f"Computing LDA projection with {self.lda_components} components")
        
        embeddings = self.embeddings
        speakers = self.process_transcript()['speaker']
        
        pipeline = Pipeline([
            ('pca', PCA(n_components=self.pca_for_lda_components)),
            ('lda', LDA(n_components=self.lda_components))
        ])
        
        pipeline.fit(embeddings, y=speakers)
        lda_vectors = pipeline.transform(embeddings)
        
        return pd.DataFrame(
            lda_vectors, 
            columns=[f'lda_{i+1}' for i in range(self.lda_components)]
        )
    
    @property
    def lda_projection(self):
        """
        Get the LDA projection DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with LDA projection columns.
        """
        return self.compute_lda_projection()
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def compute_single_lda_projection(self):
        """
        Compute single-dimension LDA projection of the embeddings based on speakers.
        
        Returns:
            pd.DataFrame: DataFrame with single LDA projection column.
        """
        if self.verbose > 0:
            print("Computing single-dimension LDA projection")
        
        embeddings = self.embeddings
        speakers = self.process_transcript()['speaker']
        
        pipeline = Pipeline([
            ('pca', PCA(n_components=self.pca_for_lda_components)),
            ('lda', LDA(n_components=1))
        ])
        
        pipeline.fit(embeddings, y=speakers)
        lda_vectors = pipeline.transform(embeddings)
        
        return pd.DataFrame(lda_vectors, columns=['single_speaker_lda'])
    
    @property
    def single_lda_projection(self):
        """
        Get the single-dimension LDA projection DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with single LDA projection column.
        """
        return self.compute_single_lda_projection()
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def compute_pca_projection(self):
        """
        Compute PCA projection of the embeddings.
        
        Returns:
            pd.DataFrame: DataFrame with PCA projection columns.
        """
        if self.verbose > 0:
            print(f"Computing PCA projection with {self.pca_components} components")
        
        embeddings = self.embeddings
        pca = PCA(n_components=self.pca_components)
        pca_vectors = pca.fit_transform(embeddings)
        
        return pd.DataFrame(
            pca_vectors, 
            columns=[f'pca_{i+1}' for i in range(self.pca_components)]
        )
    
    @property
    def pca_projection(self):
        """
        Get the PCA projection DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with PCA projection columns.
        """
        return self.compute_pca_projection()
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def compute_vader_sentiment(self):
        """
        Compute VADER sentiment analysis for the transcript text.
        
        Returns:
            pd.DataFrame: DataFrame with sentiment scores.
        """
        if self.verbose > 0:
            print("Computing VADER sentiment analysis")
        
        analyzer = SentimentIntensityAnalyzer()
        df = self.process_transcript()
        
        sentiment_scores = [analyzer.polarity_scores(text) for text in df['text']]
        return pd.DataFrame(sentiment_scores)
    
    @property
    def vader_sentiment(self):
        """
        Get the VADER sentiment analysis DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with sentiment scores.
        """
        return self.compute_vader_sentiment()
    
    @cache_this(cache='_store', key=add_extension('.parquet'))
    def compute_text2emotion(self):
        """
        Compute text2emotion analysis for the transcript text.
        
        Returns:
            pd.DataFrame: DataFrame with emotion scores.
        """
        if self.verbose > 0:
            print("Computing text2emotion analysis")
        
        try:
            import text2emotion as te
            df = self.process_transcript()
            emotion_scores = [te.get_emotion(text) for text in df['text']]
            return pd.DataFrame(emotion_scores)
        except ImportError:
            if self.verbose > 0:
                print("text2emotion package not available. Returning empty DataFrame.")
            return pd.DataFrame()
    
    @property
    def emotion_analysis(self):
        """
        Get the text2emotion analysis DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with emotion scores.
        """
        return self.compute_text2emotion()
    
    def get_complete_df(self, save=True):
        """
        Get a complete DataFrame with all computed features.
        
        Args:
            save (bool): Whether to save the DataFrame to the store.
            
        Returns:
            pd.DataFrame: Complete DataFrame with all features.
        """
        if self.verbose > 0:
            print("Creating complete DataFrame with all features")
        
        # Start with the base transcript
        df = self.process_transcript().copy()
        
        # Add projections
        df = pd.concat([df, self.tsne_projection], axis=1)
        df = pd.concat([df, self.lda_projection], axis=1)
        df = pd.concat([df, self.single_lda_projection], axis=1)
        df = pd.concat([df, self.pca_projection], axis=1)
        
        # Add sentiment analysis
        df = pd.concat([df, self.vader_sentiment], axis=1)
        
        # Add emotion analysis if available
        emotion_df = self.emotion_analysis
        if not emotion_df.empty:
            df = pd.concat([df, emotion_df], axis=1)
        
        if save:
            self.store[self.transcript_save_key] = df
        
        return df
    
    @property
    def complete_df(self):
        """
        Get a complete DataFrame with all computed features, prioritizing saved data.
        
        Returns:
            pd.DataFrame: Complete DataFrame with all features.
        """
        # Check if the complete DataFrame is already saved
        if self.transcript_save_key in self.store:
            return self.store[self.transcript_save_key]
        
        return self.get_complete_df(save=True)
    
    def get_speaker_data(self, speaker=None):
        """
        Get data for a specific speaker or all speakers.
        
        Args:
            speaker (str, optional): Speaker name. If None, returns data grouped by speaker.
            
        Returns:
            pd.DataFrame or Dict[str, pd.DataFrame]: DataFrame for the specified speaker
                or dictionary mapping speaker names to DataFrames.
        """
        df = self.complete_df
        
        if speaker is None:
            # Group by speaker and return a dictionary of DataFrames
            return {name: group for name, group in df.groupby('speaker')}
        
        # Filter for the specified speaker
        if speaker not in df['speaker'].unique():
            raise ValueError(f"Speaker '{speaker}' not found. Available speakers: {df['speaker'].unique().tolist()}")
        
        return df[df['speaker'] == speaker]
    
    def get_turn_range(self, start_turn=None, end_turn=None):
        """
        Get data for a specific range of turns.
        
        Args:
            start_turn (int, optional): Starting turn number. If None, starts from the first turn.
            end_turn (int, optional): Ending turn number. If None, ends at the last turn.
            
        Returns:
            pd.DataFrame: DataFrame for the specified turn range.
        """
        df = self.complete_df
        
        if start_turn is None:
            start_turn = 0
        if end_turn is None:
            end_turn = df['turn'].max()
        
        return df[(df['turn'] >= start_turn) & (df['turn'] <= end_turn)]
    
    def get_sentiment_summary(self, by_speaker=True):
        """
        Get a summary of sentiment scores.
        
        Args:
            by_speaker (bool): Whether to group by speaker.
            
        Returns:
            pd.DataFrame: Summary of sentiment scores.
        """
        df = self.complete_df
        
        if 'compound' not in df.columns:
            raise ValueError("Sentiment analysis not available in the complete DataFrame.")
        
        sentiment_cols = ['neg', 'neu', 'pos', 'compound']
        
        if by_speaker:
            return df.groupby('speaker')[sentiment_cols].mean()
        
        return df[sentiment_cols].mean().to_frame().T
    
    def save_all_to_store(self):
        """
        Compute and save all data to the store.
        
        Returns:
            Dict[str, Any]: Dictionary with information about saved items.
        """
        saved = {}
        
        # Get and save complete DataFrame
        saved['complete_df'] = self.get_complete_df(save=True)
        
        # Save embeddings if not already saved
        if self.embeddings_save_key not in self.store:
            embeddings_df = pd.DataFrame({'embeddings': list(self.embeddings)})
            self.store[self.embeddings_save_key] = embeddings_df
            saved['embeddings'] = embeddings_df
        
        return saved
    
    def clear_cache(self):
        """Clear the internal cache."""
        if hasattr(self, '_cache'):
            self._cache.clear()

