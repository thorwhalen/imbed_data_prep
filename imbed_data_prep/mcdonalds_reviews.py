"""McDonald's reviews data accessor"""

import os
from dataclasses import dataclass, KW_ONLY
from typing import Optional

import pandas as pd
import oa
from dol import cache_this, add_extension

from imbed.base import (
    LocalSavesMixin,
    DFLT_SAVES_DIR,
    DFLT_EMBEDDING_MODEL,
)
from imbed.data_prep import ImbedArtifactsMixin
from imbed.util import (
    planar_embeddings,
    planar_embeddings_dict_to_df,
)

data_name = 'mcdonalds_reviews'


@dataclass
class McdonaldsReviewsDacc(LocalSavesMixin, ImbedArtifactsMixin):
    """Data accessor for McDonald's reviews

    Usage:
        >>> dacc = McdonaldsReviewsDacc(datadir='.')
        >>> df = dacc.raw_data  # Loads CSV
        >>> embeddings = dacc.embeddings_df  # Computes/loads embeddings
        >>> planar = dacc.planar_embeddings  # 2D projections
        >>> clusters = dacc.clusters_df  # Cluster assignments
        >>> merged = dacc.merged_artifacts  # Everything together
    """

    name: str | None = data_name
    _: KW_ONLY
    datadir: str = '.'
    saves_dir: str | None = None
    raw_data_filename: str = 'McDonalds_Reviews_Cleaned.csv'
    verbose: int = 1
    model: str = DFLT_EMBEDDING_MODEL

    def __post_init__(self):
        # Use datadir as saves_dir if not specified
        if self.saves_dir is None:
            self.saves_dir = os.path.abspath(self.datadir)

    # @cache_this(cache='saves', key=add_extension('.csv'))
    @cache_this
    def raw_data(self):
        """Load raw CSV data

        Replaces:
            f = djoin('McDonalds_Reviews_Cleaned.csv')
            df = pd.read_csv(f)
        """
        filepath = os.path.join(self.datadir, self.raw_data_filename)
        if self.verbose:
            print(f"Loading raw data from {filepath}")
        df = pd.read_csv(filepath)
        # strip all whitespace from column names
        df.columns = [col.strip() for col in df.columns]
        return df

    @cache_this(cache='saves', key=add_extension('.parquet'))
    def embeddable(self):
        """Prepare data for embedding

        Extracts review text and adds it as 'segment' column
        """
        df = self.raw_data.copy()
        df['segment'] = df['review']
        df.index.name = 'id_'
        if self.verbose:
            print(f"Prepared {len(df)} reviews for embedding")
        return df

    @cache_this(cache='saves', key=add_extension('.parquet'))
    def embeddings_df(self):
        """Compute embeddings for reviews

        Replaces the manual pattern:
            if save_key in store:
                vectors = store[save_key]
            else:
                vectors = oa.embeddings(reviews)
                store[save_key] = vectors

        Now just call: dacc.embeddings_df
        """
        if self.verbose:
            print("Computing embeddings...")

        reviews = self.embeddable['review'].tolist()
        vectors = oa.embeddings(reviews, model=self.model)

        df = pd.DataFrame({'embedding': list(vectors)}, index=self.embeddable.index)

        if self.verbose:
            print(f"Computed {len(vectors)} embeddings")

        return df

    @cache_this(cache='saves', key=add_extension('.parquet'))
    def planar_embeddings(self):
        """Compute 2D planar embeddings for visualization"""
        if self.verbose:
            print("Computing planar embeddings...")

        embeddings_dict = self.embeddings_df['embedding'].to_dict()
        planar_dict = planar_embeddings(embeddings_dict)
        return planar_embeddings_dict_to_df(planar_dict)

    @property
    def segments(self):
        """Get segments as a dictionary (required by ImbedArtifactsMixin)"""
        df = self.embeddable
        return dict(zip(df.index.values, df.segment))

    @cache_this(cache='saves', key=add_extension('.parquet'))
    def clusters_df(self):
        """Compute cluster assignments

        Uses the parent class method which applies clustering to embeddings
        """
        if self.verbose:
            print("Computing clusters...")
        return super().clusters_df()

    @cache_this(cache='saves', key=add_extension('.parquet'))
    def merged_artifacts(self):
        """Merge all computed artifacts into one dataframe

        Combines:
        - Original review data
        - Embeddings
        - Planar (2D) embeddings
        - Cluster assignments
        """
        if self.verbose:
            print("Merging all artifacts...")

        t = self.embeddable
        t = t.merge(self.planar_embeddings, left_index=True, right_index=True)
        t = t.merge(self.clusters_df, left_index=True, right_index=True)

        return t


# -----------------------------------------------------------------------------
"""Extended McDonald's reviews data accessor with mood modeling capabilities"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections.abc import Mapping

import numpy as np
import pandas as pd
import dol
from dol import cache_this, add_extension


@dataclass
class McdonaldsReviewsMoodModeling(McdonaldsReviewsDacc):
    """Extended data accessor with mood modeling and analysis capabilities
    
    This class extends McdonaldsReviewsDacc to support:
    - Semantic attribute dataset management
    - Embeddings computation for training data
    - Model training and evaluation across categories
    - Results persistence and analysis
    
    Usage:
        >>> dacc = McdonaldsReviewsMoodModeling(
        ...     datadir='.',
        ...     semantic_attributes_path='semantic_attributes.json'
        ... )
        >>> 
        >>> # Access semantic attributes
        >>> attrs = dacc.semantic_attributes
        >>> 
        >>> # Get embeddings for a category
        >>> embeddings = dacc.category_embeddings('food_quality_presentation')
        >>> 
        >>> # Train models for all categories
        >>> results = dacc.all_category_results()
        >>> 
        >>> # Get model summary for a specific category
        >>> summary = dacc.category_model_summary('food_quality_presentation')
    """
    
    # Paths for semantic attributes and modeling
    semantic_attributes_path: str | None = 'semantic_attributes.json'
    semantic_attributes_dataset_dir: str | None = None
    embeddings_folder: str | None = None
    results_folder: str | None = None
    
    # Modeling parameters
    cv_splits: int = 30
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set default paths relative to saves_dir
        if self.semantic_attributes_dataset_dir is None:
            self.semantic_attributes_dataset_dir = os.path.join(
                self.saves_dir, 'semantic_attributes'
            )
        
        if self.embeddings_folder is None:
            self.embeddings_folder = os.path.join(
                self.saves_dir, 'embeddings'
            )
            
        if self.results_folder is None:
            self.results_folder = os.path.join(
                self.saves_dir, 'mood_modeling_results'
            )
        
        # Ensure directories exist
        for dirpath in [
            self.semantic_attributes_dataset_dir,
            self.embeddings_folder,
            self.results_folder
        ]:
            os.makedirs(dirpath, exist_ok=True)
    
    @cache_this
    def semantic_attributes(self) -> dict[str, str]:
        """Load semantic attributes from JSON file
        
        Returns dictionary mapping attribute names to descriptions.
        """
        filepath = os.path.join(self.datadir, self.semantic_attributes_path)
        
        if not os.path.exists(filepath):
            if self.verbose:
                print(f"Warning: {filepath} not found, returning empty dict")
            return {}
        
        json_store = dol.JsonFiles(os.path.dirname(filepath))
        key = os.path.basename(filepath)
        
        if self.verbose:
            print(f"Loading semantic attributes from {filepath}")
        
        return json_store[key]
    
    @property
    def _segments_store(self):
        """Internal: Create store for accessing category segments"""
        import dol
        from mood.dataset_makers import parsed_lines
        
        mk_segments_store = dol.Pipe(
            dol.TextFiles,
            dol.filt_iter(filt=lambda x: x.endswith('.txt')),
            dol.KeyCodecs.suffixed('.txt'),
            dol.wrap_kvs(value_decoder=dol.Pipe(parsed_lines, pd.DataFrame)),
        )
        
        return mk_segments_store(self.semantic_attributes_dataset_dir)
    
    @property
    def _embeddings_store(self):
        """Internal: Create store for accessing category embeddings"""
        import tabled
        return tabled.DfFiles(self.embeddings_folder)
    
    @property
    def _results_store(self):
        """Internal: Create store for accessing modeling results"""
        import dill
        
        DillFiles = dol.wrap_kvs(
            dol.Files,
            value_decoder=dill.loads,
            value_encoder=dill.dumps
        )
        
        return DillFiles(self.results_folder)
    
    def category_segments(self, category: str) -> tuple[pd.Series, pd.Series]:
        """Get segments and scores for a category
        
        Args:
            category: Name of the semantic attribute category
            
        Returns:
            Tuple of (segments, scores) as pandas Series
        """
        data_table = self._segments_store[category]
        return data_table['segment'], data_table['score']
    
    @cache_this(cache='_embeddings_store', key=lambda cat: f'{cat}.parquet')
    def category_embeddings(self, category: str) -> pd.DataFrame:
        """Compute or load embeddings for a category's segments
        
        Args:
            category: Name of the semantic attribute category
            
        Returns:
            DataFrame with embeddings column
        """
        import oa
        
        segments, _ = self.category_segments(category)
        
        if self.verbose:
            print(f"Computing embeddings for {category} ({len(segments)} segments)")
        
        embeddings = oa.embeddings(segments.to_list(), model=self.model)
        
        return pd.DataFrame({'embedding': list(embeddings)})
    
    def category_xy(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        """Get X, y arrays for model training
        
        Args:
            category: Name of the semantic attribute category
            
        Returns:
            Tuple of (X, y) as numpy arrays where:
            - X: (n_samples, embedding_dim) embedding vectors
            - y: (n_samples,) score labels
        """
        _, scores = self.category_segments(category)
        embeddings_df = self.category_embeddings(category)
        
        X = np.stack(embeddings_df['embedding'].values)
        y = np.array(scores)
        
        return X, y
    
    @cache_this(cache='_results_store', key=lambda cat: f'{cat}.p')
    def category_results(self, category: str) -> dict:
        """Train and evaluate models for a category
        
        Args:
            category: Name of the semantic attribute category
            
        Returns:
            Dictionary containing:
            - results: Single train/test split results
            - cv_results: Cross-validation results
            - summary: Performance summary
        """
        from mood.mood_modeling import MoodModelingManager
        
        if self.verbose:
            print(f"Training models for {category}")
        
        X, y = self.category_xy(category)
        manager = MoodModelingManager.from_arrays(X, y)
        
        return {
            'results': manager.train_and_evaluate(),
            'cv_results': manager.cross_validate_models(n_splits=self.cv_splits),
            'summary': manager.get_model_summary(use_cv=True),
        }
    
    def all_category_results(self) -> dict[str, dict]:
        """Train and evaluate models for all categories
        
        Returns:
            Dictionary mapping category names to their results
        """
        results = {}
        
        for category in self._segments_store:
            if self.verbose:
                print(f"\nProcessing category: {category}")
            
            results[category] = self.category_results(category)
        
        return results
    
    def category_model_summary(self, category: str) -> pd.DataFrame:
        """Get model performance summary for a category
        
        Args:
            category: Name of the semantic attribute category
            
        Returns:
            DataFrame with model performance metrics
        """
        results = self.category_results(category)
        return results['summary']
    
    def category_model_config(
        self, 
        category: str, 
        model_name: str = 'logistic_high_vs_low'
    ) -> dict:
        """Get configuration for a specific model
        
        Args:
            category: Name of the semantic attribute category
            model_name: Name of the model to get config for
            
        Returns:
            Dictionary with model configuration including class and parameters
        """
        results = self.category_results(category)
        return results['results'][model_name]['config']
    
    def train_model(
        self,
        category: str,
        model_name: str = 'logistic_high_vs_low'
    ):
        """Train and return a fitted model instance
        
        Args:
            category: Name of the semantic attribute category
            model_name: Name of the model to train
            
        Returns:
            Fitted model instance
        """
        config = self.category_model_config(category, model_name)
        X, y = self.category_xy(category)
        
        learner = config['model_class'](**config['model_params'])
        learner.fit(X, y)
        
        return learner
    
    def dataset_statistics(self) -> pd.DataFrame:
        """Get statistics about all semantic attribute datasets
        
        Returns:
            DataFrame with category names and example counts
        """
        stats = []
        
        for category in self._segments_store:
            segments, _ = self.category_segments(category)
            stats.append({
                'semantic_attribute': category,
                'number_of_examples': len(segments)
            })
        
        return pd.DataFrame(stats)
    
    def list_categories(self) -> list[str]:
        """List all available semantic attribute categories
        
        Returns:
            List of category names
        """
        return list(self._segments_store)
    
    def clear_category_cache(self, category: str):
        """Clear cached data for a category
        
        Args:
            category: Name of the semantic attribute category
        """
        # Clear embeddings
        embeddings_key = f'{category}.parquet'
        if embeddings_key in self._embeddings_store:
            del self._embeddings_store[embeddings_key]
        
        # Clear results
        results_key = f'{category}.p'
        if results_key in self._results_store:
            del self._results_store[results_key]
        
        if self.verbose:
            print(f"Cleared cache for {category}")