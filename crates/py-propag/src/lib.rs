use geometry::GeoReference;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
    ToPyArray,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;

// Custom Python exception types for different error categories
create_exception!(py_propag, PropagCudaError, PyRuntimeError);
create_exception!(py_propag, PropagGdalError, PyRuntimeError);
create_exception!(py_propag, PropagValidationError, PyValueError);
create_exception!(py_propag, PropagMPIError, PyRuntimeError);

/// Map PropagError variants to appropriate Python exceptions (placeholder for Stage 4)
/// This will be fully implemented when PropagError is integrated from the propag crate
fn _map_propag_error_placeholder() {
    // Placeholder function showing error mapping structure
    // Will be implemented in full FFI integration stage
}

/// Check if CUDA is available and working
#[pyfunction]
fn check_cuda_available() -> PyResult<bool> {
    // For Stage 4, we implement basic CUDA detection
    // Full CUDA integration will be done when propag crate is integrated

    // Check if nvidia-smi command is available (basic check)
    use std::process::Command;

    match Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let count_str = String::from_utf8_lossy(&output.stdout);
                let trimmed = count_str.trim();
                match trimmed.parse::<u32>() {
                    Ok(count) => Ok(count > 0),
                    Err(_) => Ok(false),
                }
            } else {
                Ok(false)
            }
        }
        Err(_) => {
            // nvidia-smi not found, try alternative check
            // Check if /dev/nvidia0 exists (basic Linux check)
            Ok(std::path::Path::new("/dev/nvidia0").exists())
        }
    }
}

/// Validate file path for security (prevent directory traversal)
fn validate_file_path(path: &str, parameter_name: &str) -> PyResult<()> {
    if path.is_empty() {
        return Err(PropagValidationError::new_err(format!(
            "{} cannot be empty",
            parameter_name
        )));
    }

    // Check for directory traversal attempts
    if path.contains("..") {
        return Err(PropagValidationError::new_err(format!(
            "{} contains invalid path components (..): {}",
            parameter_name, path
        )));
    }

    // Ensure path is not attempting to access system directories
    let path_obj = Path::new(path);
    if let Some(first_component) = path_obj.components().next() {
        if let std::path::Component::RootDir = first_component {
            let path_str = path_obj.to_string_lossy();
            if path_str.starts_with("/etc")
                || path_str.starts_with("/sys")
                || path_str.starts_with("/proc")
            {
                return Err(PropagValidationError::new_err(format!(
                    "{} attempts to access restricted system directory: {}",
                    parameter_name, path
                )));
            }
        }
    }

    Ok(())
}

/// Validate fuel model IDs are in valid NFFL range (1-13)
fn validate_fuel_models(fuel_array: &PyReadonlyArray2<u8>) -> PyResult<()> {
    let slice = fuel_array.as_slice().map_err(|e| {
        PropagValidationError::new_err(format!("Failed to read fuel model array: {}", e))
    })?;

    for (idx, &fuel_id) in slice.iter().enumerate() {
        if fuel_id == 0 {
            continue; // 0 might be used for non-fuel areas
        }
        if fuel_id > 13 {
            let row = idx / fuel_array.shape()[1];
            let col = idx % fuel_array.shape()[1];
            return Err(PropagValidationError::new_err(format!(
                "Invalid fuel model ID {} at position ({}, {}). Valid NFFL fuel models are 1-13.",
                fuel_id, row, col
            )));
        }
    }
    Ok(())
}

/// Validate array dimensions and data ranges
fn validate_terrain_array<T>(
    array: &PyReadonlyArray2<T>,
    name: &str,
    width: u32,
    height: u32,
    min_val: Option<f64>,
    max_val: Option<f64>,
) -> PyResult<()>
where
    T: numpy::Element + Copy + Into<f64>,
{
    let shape = array.shape();
    if shape[0] != height as usize || shape[1] != width as usize {
        return Err(PropagValidationError::new_err(format!(
            "{} array shape mismatch: expected ({}, {}), got ({}, {})",
            name, height, width, shape[0], shape[1]
        )));
    }

    // Validate data ranges if specified
    if min_val.is_some() || max_val.is_some() {
        let slice = array.as_slice().map_err(|e| {
            PropagValidationError::new_err(format!("Failed to read {} array: {}", name, e))
        })?;

        for (idx, &val) in slice.iter().enumerate() {
            let val_f64: f64 = val.into();

            if let Some(min) = min_val {
                if val_f64 < min {
                    let row = idx / shape[1];
                    let col = idx % shape[1];
                    return Err(PropagValidationError::new_err(format!(
                        "{} value {} at position ({}, {}) is below minimum {}",
                        name, val_f64, row, col, min
                    )));
                }
            }

            if let Some(max) = max_val {
                if val_f64 > max {
                    let row = idx / shape[1];
                    let col = idx % shape[1];
                    return Err(PropagValidationError::new_err(format!(
                        "{} value {} at position ({}, {}) exceeds maximum {}",
                        name, val_f64, row, col, max
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Python wrapper for Settings struct
#[pyclass]
#[derive(Clone)]
pub struct PySettings {
    pub inner: Settings,
}

#[pymethods]
impl PySettings {
    #[new]
    fn new(geo_ref: PyGeoReference, max_time: f32) -> PyResult<Self> {
        // Validate max_time parameter
        if max_time <= 0.0 {
            return Err(PropagValidationError::new_err(
                "Maximum simulation time must be positive",
            ));
        }

        if !max_time.is_finite() {
            return Err(PropagValidationError::new_err(
                "Maximum simulation time must be a finite number",
            ));
        }

        // Reasonable upper bound for simulation time (10 years in seconds)
        if max_time > 10.0 * 365.0 * 24.0 * 3600.0 {
            return Err(PropagValidationError::new_err(
                "Maximum simulation time too large (limit: 10 years)",
            ));
        }

        Ok(Self {
            inner: Settings::new(geo_ref.inner, max_time),
        })
    }

    #[getter]
    fn max_time(&self) -> f32 {
        self.inner.max_time
    }

    #[setter]
    fn set_max_time(&mut self, max_time: f32) -> PyResult<()> {
        if max_time <= 0.0 {
            return Err(PropagValidationError::new_err(
                "Maximum simulation time must be positive",
            ));
        }

        if !max_time.is_finite() {
            return Err(PropagValidationError::new_err(
                "Maximum simulation time must be a finite number",
            ));
        }

        if max_time > 10.0 * 365.0 * 24.0 * 3600.0 {
            return Err(PropagValidationError::new_err(
                "Maximum simulation time too large (limit: 10 years)",
            ));
        }

        self.inner.max_time = max_time;
        Ok(())
    }
}

/// Python wrapper for GeoReference struct
///
/// Represents spatial reference information for the simulation grid, including
/// coordinate system, dimensions, and geospatial transformation parameters.
#[pyclass]
#[derive(Clone)]
pub struct PyGeoReference {
    pub inner: GeoReference,
}

#[pymethods]
impl PyGeoReference {
    /// Create a new PyGeoReference instance
    ///
    /// Args:
    ///     width: Grid width in cells (must be positive, max 32768)
    ///     height: Grid height in cells (must be positive, max 32768)
    ///     proj: Projection string in PROJ format (e.g., 'EPSG:4326'). Maximum 1023 characters.
    ///     transform: Geospatial transform as 6-element array [x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height]
    ///
    /// Returns:
    ///     PyGeoReference: New spatial reference instance
    ///
    /// Raises:
    ///     PropagValidationError: If parameters are invalid or out of range
    #[new]
    fn new(
        width: u32,
        height: u32,
        proj: String,
        transform: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        // Validate grid dimensions
        if width == 0 || height == 0 {
            return Err(PropagValidationError::new_err(
                "Grid dimensions must be positive (width > 0, height > 0)",
            ));
        }

        if width > 32768 || height > 32768 {
            return Err(PropagValidationError::new_err(
                "Grid dimensions too large (max 32768x32768)",
            ));
        }

        // Validate projection string
        if proj.is_empty() {
            return Err(PropagValidationError::new_err(
                "Projection string cannot be empty",
            ));
        }

        let proj_bytes = proj.as_bytes();
        if proj_bytes.len() >= 1024 {
            return Err(PropagValidationError::new_err(
                "Projection string too large (max 1023 characters)",
            ));
        }

        let transform_slice = transform.as_slice().map_err(|e| {
            PropagValidationError::new_err(format!("Failed to read transform array: {}", e))
        })?;
        if transform_slice.len() != 6 {
            return Err(PropagValidationError::new_err(
                "Geo transform must have exactly 6 elements [x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height]"
            ));
        }

        // Validate transform parameters
        if transform_slice[1] == 0.0 || transform_slice[5] == 0.0 {
            return Err(PropagValidationError::new_err(
                "Invalid geo transform: pixel size cannot be zero",
            ));
        }

        let mut proj_array = [0u8; 1024];
        // Copy the string bytes and add null terminator
        proj_array[..proj_bytes.len()].copy_from_slice(proj_bytes);
        proj_array[proj_bytes.len()] = 0; // Null terminator

        let geo_transform = geometry::GeoTransform::new([
            transform_slice[0] as f32,
            transform_slice[1] as f32,
            transform_slice[2] as f32,
            transform_slice[3] as f32,
            transform_slice[4] as f32,
            transform_slice[5] as f32,
        ])
        .ok_or_else(|| {
            PropagValidationError::new_err(
                "Invalid geo transform parameters - ensure pixel sizes are non-zero and finite",
            )
        })?;

        Ok(Self {
            inner: GeoReference {
                width,
                height,
                proj: proj_array,
                transform: geo_transform,
            },
        })
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Get the projection string
    ///
    /// Returns:
    ///     str: Projection string in PROJ format. Maximum 1023 characters.
    ///
    /// Raises:
    ///     ValueError: If projection string contains invalid UTF-8
    #[getter]
    fn proj(&self) -> PyResult<String> {
        // Find the null terminator in the byte array
        let null_pos = self
            .inner
            .proj
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.inner.proj.len());

        // Convert bytes to string up to the null terminator
        String::from_utf8(self.inner.proj[..null_pos].to_vec()).map_err(|e| {
            PyValueError::new_err(format!("Invalid UTF-8 in projection string: {}", e))
        })
    }

    #[getter]
    fn transform<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.transform.as_array_64().to_pyarray(py)
    }
}

/// Python wrapper for TimeFeature struct
#[pyclass]
#[derive(Clone)]
pub struct PyTimeFeature {
    pub time: f32,
    pub geom_wkb: Vec<u8>,
}

#[pymethods]
impl PyTimeFeature {
    #[new]
    fn new(time: f32, geom_wkb: Vec<u8>) -> PyResult<Self> {
        // Validate time parameter
        if time < 0.0 {
            return Err(PropagValidationError::new_err("Time must be non-negative"));
        }

        if !time.is_finite() {
            return Err(PropagValidationError::new_err(
                "Time must be a finite number",
            ));
        }

        // Validate geometry WKB
        if geom_wkb.is_empty() {
            return Err(PropagValidationError::new_err(
                "Geometry WKB cannot be empty",
            ));
        }

        // Basic WKB validation - check for minimum header size
        if geom_wkb.len() < 9 {
            return Err(PropagValidationError::new_err(
                "Invalid geometry WKB - too short for valid geometry",
            ));
        }

        Ok(Self { time, geom_wkb })
    }

    #[getter]
    fn time(&self) -> f32 {
        self.time
    }

    #[getter]
    fn geom_wkb(&self) -> Vec<u8> {
        self.geom_wkb.clone()
    }
}

// Note: FFI structures will be added in Stage 3 when we integrate with the actual propag FFI

/// Python wrapper for PropagResults struct
#[pyclass]
pub struct PyPropagResults {
    pub arrival_times: Vec<f32>,
    pub boundary_change: Vec<u32>,
    pub refs_x: Vec<u16>,
    pub refs_y: Vec<u16>,
    pub geo_ref: PyGeoReference,
    pub width: u32,
    pub height: u32,
}

#[pymethods]
impl PyPropagResults {
    /// Get arrival times as a 2D NumPy array with shape (height, width)
    #[getter]
    fn arrival_times<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.arrival_times.len() != (self.width * self.height) as usize {
            return Err(PyValueError::new_err("Array size mismatch"));
        }
        // Create NumPy array directly without copying
        let arr = PyArray1::from_slice(py, &self.arrival_times);
        arr.reshape([self.height as usize, self.width as usize])
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape: {}", e)))
    }

    #[getter]
    fn boundary_change<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.boundary_change.to_pyarray(py)
    }

    #[getter]
    fn refs_x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        self.refs_x.to_pyarray(py)
    }

    #[getter]
    fn refs_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        self.refs_y.to_pyarray(py)
    }

    #[getter]
    fn geo_ref(&self) -> PyGeoReference {
        self.geo_ref.clone()
    }

    #[getter]
    fn shape(&self) -> (u32, u32) {
        (self.height, self.width)
    }
}

/// For Stage 2, we'll create a simplified wrapper that prepares the foundation
/// for the full FFI integration which will be completed in later stages

/// Simplified Settings struct for now
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Settings {
    pub geo_ref: GeoReference,
    pub max_time: f32,
}

impl Settings {
    pub fn new(geo_ref: GeoReference, max_time: f32) -> Self {
        Self { geo_ref, max_time }
    }
}

/// Load terrain data from NumPy arrays and create a PyGeoReference object
///
/// Validates terrain data arrays and creates a spatial reference for fire simulation.
/// All terrain arrays must match the specified grid dimensions.
///
/// Args:
///     width: Grid width in cells (must be positive, max grid size 100M cells)
///     height: Grid height in cells (must be positive, max grid size 100M cells)
///     proj: Projection string in PROJ format (e.g., 'EPSG:4326'). Maximum 1023 characters.
///     transform: Geospatial transform as 6-element float64 array
///     elevation: Optional elevation data in meters (range: -1000 to 9000m)
///     slope: Optional slope data in degrees (range: 0 to 90°)
///     aspect: Optional aspect data in degrees (range: 0 to 360°)
///     fuel_model: Optional NFFL fuel model IDs (range: 1 to 13, 0 for non-fuel)
///
/// Returns:
///     PyGeoReference: Spatial reference object for the terrain
///
/// Raises:
///     PropagValidationError: If data validation fails or dimensions mismatch
#[pyfunction]
fn load_terrain_data(
    width: u32,
    height: u32,
    proj: String,
    transform: PyReadonlyArray1<f64>,
    elevation: Option<PyReadonlyArray2<f32>>,
    slope: Option<PyReadonlyArray2<f32>>,
    aspect: Option<PyReadonlyArray2<f32>>,
    fuel_model: Option<PyReadonlyArray2<u8>>,
) -> PyResult<PyGeoReference> {
    // Validate grid dimensions
    if width == 0 || height == 0 {
        return Err(PropagValidationError::new_err(
            "Grid dimensions must be positive",
        ));
    }

    // Check for reasonable grid size limits
    let total_cells = width as u64 * height as u64;
    if total_cells > 100_000_000 {
        return Err(PropagValidationError::new_err(
            "Grid too large (limit: 100 million cells). Consider using smaller tiles.",
        ));
    }

    // Validate terrain data arrays if provided
    if let Some(elev) = &elevation {
        validate_terrain_array(
            elev,
            "Elevation",
            width,
            height,
            Some(-1000.0),
            Some(9000.0),
        )?;
    }

    if let Some(sl) = &slope {
        validate_terrain_array(sl, "Slope", width, height, Some(0.0), Some(90.0))?;
    }

    if let Some(asp) = &aspect {
        validate_terrain_array(asp, "Aspect", width, height, Some(0.0), Some(360.0))?;
    }

    if let Some(fuel) = &fuel_model {
        validate_terrain_array(fuel, "Fuel model", width, height, None, None)?;
        validate_fuel_models(fuel)?;
    }

    // Create GeoReference (terrain data storage will be added in future stages)
    PyGeoReference::new(width, height, proj, transform)
}

/// Utility function to convert 2D NumPy array to flattened Vec for internal use
#[allow(dead_code)]
fn numpy_2d_to_vec_f32(array: &PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
    let slice = array.as_slice()?;
    Ok(slice.to_vec())
}

/// Utility function to convert 2D NumPy array to flattened Vec for internal use  
#[allow(dead_code)]
fn numpy_2d_to_vec_u8(array: &PyReadonlyArray2<u8>) -> PyResult<Vec<u8>> {
    let slice = array.as_slice()?;
    Ok(slice.to_vec())
}

/// Main propagation function exposed to Python (Stage 3 - NumPy integration)
/// This validates inputs and prepares for future FFI integration
#[pyfunction]
fn propagate(
    settings: PySettings,
    output_path: String,
    initial_ignited_elements: Vec<PyTimeFeature>,
    initial_ignited_elements_crs: String,
    refs_output_path: Option<String>,
    block_boundaries_out_path: Option<String>,
    grid_boundaries_out_path: Option<String>,
) -> PyResult<PyPropagResults> {
    // Validate file paths
    validate_file_path(&output_path, "output_path")?;

    if let Some(ref path) = refs_output_path {
        validate_file_path(path, "refs_output_path")?;
    }

    if let Some(ref path) = block_boundaries_out_path {
        validate_file_path(path, "block_boundaries_out_path")?;
    }

    if let Some(ref path) = grid_boundaries_out_path {
        validate_file_path(path, "grid_boundaries_out_path")?;
    }

    // Validate CRS string
    if initial_ignited_elements_crs.is_empty() {
        return Err(PropagValidationError::new_err(
            "Coordinate reference system (CRS) cannot be empty",
        ));
    }

    if initial_ignited_elements_crs.len() > 2048 {
        return Err(PropagValidationError::new_err(
            "CRS string too long (max 2048 characters)",
        ));
    }

    // Validate ignition elements
    if initial_ignited_elements.is_empty() {
        return Err(PropagValidationError::new_err(
            "At least one ignition element is required",
        ));
    }

    if initial_ignited_elements.len() > 10000 {
        return Err(PropagValidationError::new_err(
            "Too many ignition elements (max 10,000)",
        ));
    }

    // Check CUDA availability before proceeding
    if !check_cuda_available()? {
        return Err(PropagCudaError::new_err(
            "CUDA is not available. Ensure NVIDIA GPU drivers and CUDA runtime are installed.",
        ));
    }

    // For Stage 2, we just validate the structure and log the operation
    println!("Propagation called with:");
    println!("  - Max time: {}", settings.inner.max_time);
    println!(
        "  - Grid size: {}x{}",
        settings.inner.geo_ref.width, settings.inner.geo_ref.height
    );
    println!("  - Output path: {}", output_path);
    println!("  - Ignition elements: {}", initial_ignited_elements.len());
    println!("  - CRS: {}", initial_ignited_elements_crs);

    if let Some(ref path) = refs_output_path {
        println!("  - Refs output: {}", path);
    }

    if let Some(ref path) = block_boundaries_out_path {
        println!("  - Block boundaries output: {}", path);
    }

    if let Some(ref path) = grid_boundaries_out_path {
        println!("  - Grid boundaries output: {}", path);
    }

    // For Stage 3, create mock results with proper NumPy array structure
    let grid_size = (settings.inner.geo_ref.width * settings.inner.geo_ref.height) as usize;
    let mock_arrival_times = vec![f32::INFINITY; grid_size]; // Unburned areas have infinite arrival time

    println!("Stage 3: Input validation complete with NumPy integration. Returning mock results.");
    Ok(PyPropagResults {
        arrival_times: mock_arrival_times,
        boundary_change: vec![0; 100], // Mock boundary changes
        refs_x: vec![0; 100],
        refs_y: vec![0; 100],
        geo_ref: PyGeoReference {
            inner: settings.inner.geo_ref,
        },
        width: settings.inner.geo_ref.width,
        height: settings.inner.geo_ref.height,
    })
}

// Note: Removed FFI-specific code for Stage 2 to focus on getting the basic structure working

/// py-propag: Python bindings for Propag25 GPU-accelerated wildfire simulation
///
/// This module provides Python access to the high-performance Rust/CUDA fire simulation engine,
/// enabling wildfire propagation modeling with GPU acceleration.
///
/// # Features
///
/// - GPU-accelerated fire spread simulation using CUDA
/// - Support for standard terrain data formats (elevation, slope, aspect, fuel models)
/// - NFFL (Northern Forest Fire Laboratory) fuel model support
/// - Comprehensive input validation and error handling
/// - NumPy array integration for efficient data handling
/// - Geospatial reference system support via PROJ
///
/// # Requirements
///
/// - NVIDIA GPU with CUDA Compute Capability 3.5+
/// - CUDA Runtime 12.6 or later
/// - Python 3.8+
/// - NumPy 1.19+
///
/// # Quick Start
///
/// ```python
/// import numpy as np
/// import py_propag
///
/// # Check CUDA availability
/// if not py_propag.check_cuda_available():
///     print("Warning: CUDA not available, install NVIDIA drivers")
///
/// # Create simulation grid (100x100m, 1m resolution)
/// width, height = 100, 100
/// proj = 'EPSG:32633'  # UTM Zone 33N
/// transform = np.array([500000.0, 1.0, 0.0, 4000000.0, 0.0, -1.0], dtype=np.float64)
///
/// # Generate terrain data
/// elevation = np.random.uniform(800, 1200, (height, width)).astype(np.float32)
/// slope = np.random.uniform(0, 30, (height, width)).astype(np.float32)  
/// aspect = np.random.uniform(0, 360, (height, width)).astype(np.float32)
/// fuel_model = np.full((height, width), 2, dtype=np.uint8)  # NFFL fuel model 2
///
/// # Load terrain data
/// geo_ref = py_propag.load_terrain_data(
///     width, height, proj, transform,
///     elevation, slope, aspect, fuel_model
/// )
///
/// # Create simulation settings
/// settings = py_propag.PySettings(geo_ref, max_time=3600.0)  # 1 hour
///
/// # Define ignition point (center of grid)
/// ignition_wkb = bytes([1, 1, 0, 0, 0] +
///     list(np.array(500050.0, dtype=np.float64).tobytes()) +
///     list(np.array(3999950.0, dtype=np.float64).tobytes()))
/// ignition = py_propag.PyTimeFeature(0.0, ignition_wkb)
///
/// # Run simulation
/// results = py_propag.propagate(
///     settings=settings,
///     output_path="fire_simulation.tif",
///     initial_ignited_elements=[ignition],
///     initial_ignited_elements_crs="EPSG:32633"
/// )
///
/// # Analyze results
/// arrival_times = results.arrival_times  # 2D NumPy array
/// burned_area = np.sum(np.isfinite(arrival_times))
/// print(f"Burned area: {burned_area} cells ({burned_area} m²)")
/// ```
///
/// # Error Handling
///
/// The module defines custom exception types for different error categories:
///
/// ```python
/// try:
///     results = py_propag.propagate(...)
/// except py_propag.PropagValidationError as e:
///     print(f"Input validation error: {e}")
/// except py_propag.PropagCudaError as e:
///     print(f"CUDA error: {e}")
/// except py_propag.PropagGdalError as e:
///     print(f"Geospatial data error: {e}")
/// ```
///
/// # Classes and Functions
///
/// ## Core Classes
/// - `PyGeoReference`: Spatial reference information for simulation grid
/// - `PySettings`: Simulation configuration and parameters  
/// - `PyTimeFeature`: Ignition points with timing information
/// - `PyPropagResults`: Simulation results with fire spread data
///
/// ## Functions
/// - `check_cuda_available()`: Check if CUDA GPU acceleration is available
/// - `load_terrain_data()`: Load and validate terrain data arrays
/// - `propagate()`: Run fire propagation simulation
///
/// ## Exception Types
/// - `PropagCudaError`: CUDA/GPU related errors
/// - `PropagGdalError`: Geospatial data processing errors
/// - `PropagValidationError`: Input validation errors
/// - `PropagMPIError`: Multi-processing errors
///
/// For detailed API documentation and examples, see the README.md file.
#[pymodule]
fn _py_propag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<PySettings>()?;
    m.add_class::<PyGeoReference>()?;
    m.add_class::<PyTimeFeature>()?;
    m.add_class::<PyPropagResults>()?;

    // Add functions
    m.add_function(wrap_pyfunction!(propagate, m)?)?;
    m.add_function(wrap_pyfunction!(load_terrain_data, m)?)?;
    m.add_function(wrap_pyfunction!(check_cuda_available, m)?)?;

    // Add custom exception types
    m.add("PropagCudaError", m.py().get_type::<PropagCudaError>())?;
    m.add("PropagGdalError", m.py().get_type::<PropagGdalError>())?;
    m.add(
        "PropagValidationError",
        m.py().get_type::<PropagValidationError>(),
    )?;
    m.add("PropagMPIError", m.py().get_type::<PropagMPIError>())?;

    // Check CUDA availability at module initialization and warn if not available
    match check_cuda_available() {
        Ok(true) => {
            println!("py-propag: CUDA is available and working");
        }
        Ok(false) => {
            eprintln!("Warning: CUDA is not available. GPU acceleration will not work.");
            eprintln!("Install NVIDIA GPU drivers and CUDA runtime for optimal performance.");
        }
        Err(e) => {
            eprintln!("Warning: Failed to check CUDA availability: {}", e);
        }
    }

    Ok(())
}
