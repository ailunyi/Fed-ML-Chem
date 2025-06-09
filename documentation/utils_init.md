# utils/__init__.py

## Overview
This file is the Python package initialization file for the `utils` package. It imports and re-exports all the functions and classes from the other utility files in the package, making them directly accessible when importing from the `utils` package.

## Content
```python
from .common import *
from .data_setup import *
from .engine import *
```

## Purpose
The purpose of this file is to simplify imports in the notebooks and other Python files that use the utility functions. Instead of having to import functions from specific modules within the `utils` package, users can import them directly from the `utils` package.

## Usage Examples

### Without utils/__init__.py
```python
from utils.common import choice_device, classes_string
from utils.data_setup import load_datasets
from utils.engine import train, test
```

### With utils/__init__.py
```python
from utils import choice_device, classes_string, load_datasets, train, test
```

This simplifies imports and makes the code more concise and readable, especially when multiple utility functions are used across different modules.

## Package Structure
The `utils` package contains the following modules:
- `common.py`: Common utility functions for file operations, device selection, dataset utilities, visualization, and model parameter handling
- `data_setup.py`: Functions for data preprocessing, normalization, splitting, and loading
- `engine.py`: Functions for training and testing models

All functions and classes from these modules are made available at the package level through this `__init__.py` file.