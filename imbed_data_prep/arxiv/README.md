# ArXiv

Data preparation for ArXiv papers.

## Status

This module has been migrated to the standalone
[`xv`](https://pypi.org/project/xv/) package on PyPI.
The module here is a thin wrapper that re-exports from `xv`.

## Data source

[ArXiv](https://arxiv.org/) is an open-access repository of scientific
papers in physics, mathematics, computer science, and related fields, hosted
by Cornell University.

## Usage

```bash
pip install xv
```

```python
from imbed_data_prep.arxiv import ...  # delegates to xv
```

## Files in this directory

| File | Description |
|---|---|
| `__init__.py` | Wrapper module importing from the `xv` package |
