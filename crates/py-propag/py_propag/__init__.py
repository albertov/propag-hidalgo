"""
py-propag - Python bindings for the Propag GPU-accelerated wildfire propagation simulation system.
"""

# Import everything from the Rust extension module
from ._py_propag import (
    # Classes
    PySettings,
    PyGeoReference,
    PyTimeFeature,
    PyPropagResults,
    
    # Functions
    propagate,
    load_terrain_data,
    check_cuda_available,
    
    # Exceptions
    PropagCudaError,
    PropagGdalError,
    PropagValidationError,
    PropagMPIError,
)

# Re-export with cleaner names (without the Py prefix for classes)
Settings = PySettings
GeoReference = PyGeoReference
TimeFeature = PyTimeFeature
PropagResults = PyPropagResults

__all__ = [
    # Classes (both names for compatibility)
    'Settings', 'PySettings',
    'GeoReference', 'PyGeoReference',
    'TimeFeature', 'PyTimeFeature',
    'PropagResults', 'PyPropagResults',
    
    # Functions
    'propagate',
    'load_terrain_data',
    'check_cuda_available',
    
    # Exceptions
    'PropagCudaError',
    'PropagGdalError',
    'PropagValidationError',
    'PropagMPIError',
]

__version__ = "0.1.0"