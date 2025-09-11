# Shared Utilities - Fertility Reconstruction

This directory contains shared utilities and helper functions used by both the data pipeline and modeling pipeline.

## Directory Structure

```
shared/
├── utils/            # Utility modules
│   ├── ee_helpers.py      # Google Earth Engine helpers
│   ├── geometry.py        # Geometric operations
│   └── __init__.py        # Package initialization
└── README.md        # This file
```

## Utilities

### `utils/ee_helpers.py`
Helper functions for Google Earth Engine operations:
- Authentication handling
- Common GEE data processing functions
- Error handling and retry logic

### `utils/geometry.py` 
Geometric operations and spatial utilities:
- Coordinate system transformations
- Spatial indexing and operations
- Region of interest management

## Usage

Both pipelines can import these utilities:

```python
# From data pipeline
from shared.utils.ee_helpers import authenticate_ee
from shared.utils.geometry import convert_coordinates

# From modeling pipeline  
from shared.utils.geometry import pixel_to_coordinates
```

## Adding New Utilities

When adding new shared functions:

1. **Create appropriate module** in `utils/`
2. **Add clear documentation** and type hints
3. **Include unit tests** (when test framework is set up)
4. **Update this README** with new functions

## Dependencies

Shared utilities should have minimal dependencies and be compatible with both pipeline environments.
