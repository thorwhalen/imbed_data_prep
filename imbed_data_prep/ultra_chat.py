"""Ultra Chat data preparation module."""

from functools import lru_cache


def get_raw_data():
    """Get the Ultra Chat data."""
    from datasets import load_dataset

    return load_dataset('stingning/ultrachat')
