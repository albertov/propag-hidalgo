# py-propag: Python Bindings for Propag25 Fire Simulation

Python bindings for the Propag25 GPU-accelerated wildfire propagation simulation system, part of the HiDALGO2 initiative.

## Overview

py-propag provides Python access to the high-performance Rust/CUDA fire simulation engine, enabling wildfire propagation modeling with GPU acceleration. The library handles terrain data, fuel models, weather conditions, and ignition patterns to simulate fire spread over time.

## Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5+ 
- **CUDA Runtime**: CUDA 12.6 or later
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+) or Windows 10/11
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large simulations
- **Python**: Python 3.8 or later

### GPU Requirements
The library requires an NVIDIA GPU for optimal performance:
- **Minimum**: GTX 1060 / RTX 2060 / Tesla K40
- **Recommended**: RTX 3070+ / A100 / V100 for large-scale simulations
- **VRAM**: Minimum 4GB, 8GB+ recommended

### Python Dependencies
- `numpy >= 1.19.0`
- `pytest >= 6.0` (for testing)

## Installation

### From Source (Recommended)

1. **Install Rust and CUDA dependencies:**
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   
   # Install CUDA (Ubuntu example)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.0-560.28.03-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.0-560.28.03-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-6
   ```

2. **Install Python dependencies:**
   ```bash
   pip install numpy maturin
   ```

3. **Build and install py-propag:**
   ```bash
   # Clone the repository
   git clone https://github.com/your-org/propag25.git
   cd propag25/crates/py-propag
   
   # Build and install with maturin
   maturin develop --release
   ```

### Verify Installation

Check that CUDA is available and the module imports correctly:

```python
import py_propag
import numpy as np

# Check CUDA availability
cuda_available = py_propag.check_cuda_available()
print(f"CUDA available: {cuda_available}")

if not cuda_available:
    print("Warning: CUDA not detected. Install NVIDIA drivers and CUDA runtime.")
```

## Quick Start

### Basic Fire Simulation

```python
import numpy as np
import py_propag

# 1. Define simulation grid (100x100 meters, 1m resolution)
width, height = 100, 100
proj = np.array(b'EPSG:32633\x00', dtype=np.uint8)  # UTM Zone 33N
transform = np.array([
    500000.0,  # X origin (UTM easting)
    1.0,       # Pixel width (1 meter)
    0.0,       # X rotation
    4000000.0, # Y origin (UTM northing)  
    0.0,       # Y rotation
    -1.0       # Pixel height (-1 meter, north-up)
], dtype=np.float64)

# 2. Create terrain data
elevation = np.random.uniform(800, 1200, (height, width)).astype(np.float32)
slope = np.random.uniform(0, 30, (height, width)).astype(np.float32)
aspect = np.random.uniform(0, 360, (height, width)).astype(np.float32)
fuel_model = np.full((height, width), 2, dtype=np.uint8)  # NFFL fuel model 2

# 3. Load terrain data
geo_ref = py_propag.load_terrain_data(
    width=width,
    height=height, 
    proj=proj,
    transform=transform,
    elevation=elevation,
    slope=slope,
    aspect=aspect,
    fuel_model=fuel_model
)

# 4. Create simulation settings
max_time = 3600.0  # 1 hour simulation
settings = py_propag.PySettings(geo_ref, max_time)

# 5. Define ignition point
ignition_wkb = bytes([
    1,  # Little endian
    1, 0, 0, 0,  # Point geometry type
    # Coordinates in UTM (center of grid)
    *np.array(500050.0, dtype=np.float64).tobytes(),  # X
    *np.array(3999950.0, dtype=np.float64).tobytes()  # Y
])

ignition = py_propag.PyTimeFeature(
    time=0.0,  # Ignition at start
    geom_wkb=ignition_wkb
)

# 6. Run simulation
results = py_propag.propagate(
    settings=settings,
    output_path="fire_simulation.tif",
    initial_ignited_elements=[ignition],
    initial_ignited_elements_crs="EPSG:32633"
)

# 7. Analyze results
arrival_times = results.arrival_times  # 2D NumPy array
burned_mask = np.isfinite(arrival_times)
burned_area = np.sum(burned_mask)  # Number of burned cells

print(f"Simulation completed:")
print(f"  - Burned area: {burned_area} cells ({burned_area} m²)")
print(f"  - Grid shape: {arrival_times.shape}")
print(f"  - Max arrival time: {np.max(arrival_times[burned_mask]):.1f} seconds")
```

### Working with Real Terrain Data

```python
import numpy as np
import py_propag
from osgeo import gdal  # Optional: for loading GeoTIFF files

# Load terrain data from GeoTIFF files (requires GDAL)
def load_geotiff(filename):
    """Load a GeoTIFF file as NumPy array with geo-reference."""
    dataset = gdal.Open(filename)
    array = dataset.ReadAsArray().astype(np.float32)
    
    # Get geo-transform and projection
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    return array, geo_transform, projection

# Example: Load real terrain data
try:
    elevation, geo_transform, projection = load_geotiff("elevation.tif")
    slope, _, _ = load_geotiff("slope.tif") 
    aspect, _, _ = load_geotiff("aspect.tif")
    fuel_model, _, _ = load_geotiff("fuel_models.tif")
    
    # Convert to required formats
    height, width = elevation.shape
    proj_bytes = np.array(projection.encode('utf-8') + b'\x00', dtype=np.uint8)
    transform_array = np.array(geo_transform, dtype=np.float64)
    
    # Load terrain
    geo_ref = py_propag.load_terrain_data(
        width=width,
        height=height,
        proj=proj_bytes,
        transform=transform_array,
        elevation=elevation,
        slope=slope,
        aspect=aspect,
        fuel_model=fuel_model.astype(np.uint8)
    )
    
    print(f"Loaded terrain: {width}x{height} cells")
    
except ImportError:
    print("GDAL not available. Install with: pip install gdal")
except Exception as e:
    print(f"Error loading terrain data: {e}")
```

## API Reference

### Classes

#### `PyGeoReference`
Represents spatial reference information for the simulation grid.

```python
geo_ref = py_propag.PyGeoReference(
    width: int,           # Grid width in cells
    height: int,          # Grid height in cells  
    proj: np.ndarray,     # Projection string as uint8 array
    transform: np.ndarray # 6-element geo-transform array
)

# Properties
geo_ref.width      # Grid width
geo_ref.height     # Grid height
geo_ref.proj       # Projection array
geo_ref.transform  # Geo-transform array
```

#### `PySettings`
Simulation configuration and parameters.

```python
settings = py_propag.PySettings(
    geo_ref: PyGeoReference,  # Spatial reference
    max_time: float           # Maximum simulation time (seconds)
)

# Properties  
settings.max_time          # Get/set maximum time
```

#### `PyTimeFeature`
Represents an ignition point with timing information.

```python
feature = py_propag.PyTimeFeature(
    time: float,        # Ignition time (seconds)
    geom_wkb: bytes     # Geometry in WKB format
)

# Properties
feature.time        # Ignition time
feature.geom_wkb    # Geometry bytes
```

#### `PyPropagResults`
Simulation results containing fire spread information.

```python
# Properties (read-only)
results.arrival_times    # 2D array: fire arrival times
results.boundary_change  # 1D array: boundary change indices
results.refs_x          # 1D array: reference X coordinates  
results.refs_y          # 1D array: reference Y coordinates
results.geo_ref         # Spatial reference information
results.shape           # Grid dimensions (height, width)
```

### Functions

#### `check_cuda_available() -> bool`
Check if CUDA is available for GPU acceleration.

```python
if py_propag.check_cuda_available():
    print("GPU acceleration available")
else:
    print("Running on CPU (slower)")
```

#### `load_terrain_data(...) -> PyGeoReference`
Load and validate terrain data arrays.

```python
geo_ref = py_propag.load_terrain_data(
    width: int,                    # Grid width
    height: int,                   # Grid height
    proj: np.ndarray,              # Projection (uint8)
    transform: np.ndarray,         # Geo-transform (float64)
    elevation: np.ndarray = None,  # Elevation (float32)
    slope: np.ndarray = None,      # Slope degrees (float32)
    aspect: np.ndarray = None,     # Aspect degrees (float32)  
    fuel_model: np.ndarray = None  # NFFL fuel IDs (uint8)
)
```

#### `propagate(...) -> PyPropagResults`
Run fire propagation simulation.

```python
results = py_propag.propagate(
    settings: PySettings,                    # Simulation settings
    output_path: str,                        # Output file path
    initial_ignited_elements: List[PyTimeFeature],  # Ignition points
    initial_ignited_elements_crs: str,       # CRS of ignition points
    refs_output_path: str = None,            # Optional reference output
    block_boundaries_out_path: str = None,   # Optional block boundaries
    grid_boundaries_out_path: str = None     # Optional grid boundaries
)
```

### Exception Types

The library defines custom exception types for different error categories:

- `PropagCudaError`: CUDA/GPU related errors
- `PropagGdalError`: Geospatial data processing errors  
- `PropagValidationError`: Input validation errors
- `PropagMPIError`: Multi-processing errors

## Error Handling

```python
import py_propag

try:
    # Check CUDA first
    if not py_propag.check_cuda_available():
        raise py_propag.PropagCudaError("CUDA not available")
    
    # Run simulation
    results = py_propag.propagate(...)
    
except py_propag.PropagValidationError as e:
    print(f"Input validation error: {e}")
    
except py_propag.PropagCudaError as e:
    print(f"CUDA error: {e}")
    print("Make sure NVIDIA drivers and CUDA runtime are installed")
    
except py_propag.PropagGdalError as e:
    print(f"Geospatial data error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Data Formats

### Terrain Data Requirements

- **Elevation**: Float32 array, meters above sea level (-1000 to 9000m)
- **Slope**: Float32 array, degrees (0-90°)
- **Aspect**: Float32 array, degrees (0-360°, 0=North)
- **Fuel Models**: Uint8 array, NFFL fuel model IDs (1-13, 0=non-fuel)

### Coordinate Systems

The library supports any coordinate system definable in PROJ format:
- Geographic coordinates (EPSG:4326)
- UTM projections (EPSG:32601-32660, EPSG:32701-32760)
- State Plane coordinates
- Custom PROJ strings

### File Outputs

Simulation results are saved as GeoTIFF files:
- **Arrival times**: Float32 raster with fire arrival times
- **References**: Point coordinates for boundary tracking
- **Boundaries**: Vector data for fire perimeter evolution

## Performance Tips

### GPU Memory Management
```python
# Monitor GPU memory usage for large simulations
def check_gpu_memory():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory: {info.used/1024**3:.1f}GB used / {info.total/1024**3:.1f}GB total")
    except ImportError:
        print("Install pynvml for GPU memory monitoring: pip install pynvml")

# For large grids, consider tiling
def process_large_terrain(elevation, max_size=10000):
    """Process large terrain in tiles to manage memory."""
    if elevation.shape[0] <= max_size and elevation.shape[1] <= max_size:
        return process_single_tile(elevation)
    
    # Split into tiles and process separately
    # ... tiling logic here ...
```

### Optimization Guidelines
- Use float32 for terrain data (sufficient precision, better performance)
- Keep grid sizes under 10,000x10,000 cells for single GPU
- Pre-validate input data to avoid runtime errors
- Use appropriate fuel models for your region

## Troubleshooting

### Common Issues

**CUDA not detected:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation  
nvcc --version

# Verify device files exist
ls -la /dev/nvidia*
```

**Import errors:**
```bash
# Rebuild the module
cd crates/py-propag
maturin develop --release

# Check Python path
python -c "import sys; print(sys.path)"
```

**Memory errors:**
- Reduce grid size or use tiling for large simulations
- Close other GPU applications
- Use smaller data types where possible

**Validation errors:**
- Check terrain data ranges and types
- Verify coordinate system specifications
- Ensure fuel model IDs are valid (1-13)

## Contributing

This library is part of the Propag25 project. See the main repository for contribution guidelines and development setup.

## License

This project is funded by the European Union under the HiDALGO2 initiative. See LICENSE file for details.

## Support

For technical support and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [documentation-url]
- HiDALGO2 Project: [project-url]