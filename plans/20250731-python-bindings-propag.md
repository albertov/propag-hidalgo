# Python Bindings for Propag Crate Implementation Plan

**Date**: 2025-07-31  
**Feature**: Implement Python bindings for the propag crate using the maturin tool

## Executive Summary

This document outlines the implementation plan for creating Python bindings for the propag crate, a GPU-accelerated wildfire propagation simulation system. The bindings will be built using maturin and PyO3, leveraging the existing py-propag crate structure that currently contains only placeholder code.

## Feature Analysis

The requested feature involves:
- Creating Python bindings for the propag crate (GPU-accelerated fire propagation engine)
- Using maturin as the build tool for Rust-based Python extensions
- Exposing propag's CUDA-based fire spread calculations to Python users
- Integrating with PyO3 for Rust-Python interoperability

## Repository Context

### Current State
- **Existing Infrastructure**: A `py-propag` crate exists at `crates/py-propag/` with:
  - Basic maturin setup (`pyproject.toml`)
  - PyO3 dependencies configured in `Cargo.toml`
  - Only placeholder implementation (`sum_as_string` function)
  - Nix build configuration (`py-propag.nix`)

- **Core propag Architecture**:
  - GPU-accelerated wildfire simulation using CUDA
  - Complex data structures: `Settings`, `Terrain`, `Fire`, `PropagResults`
  - Existing C FFI interface: `FFIPropagation_run`
  - Heavy dependencies: CUDA runtime, GDAL, MPI
  - CUDA kernels located in `crates/propag/src/propag.cu`

- **Integration Points**:
  - firelib crate provides fire behavior models (Rothermel model)
  - dmoist crate calculates fuel moisture
  - geometry crate handles spatial calculations

### Key Challenges
1. CUDA runtime requirements and GPU availability
2. Complex memory layouts for GPU compatibility
3. Geospatial data handling (GDAL integration)
4. Large array management for terrain grids
5. Error propagation from multiple subsystems (CUDA, GDAL, MPI)

## Implementation Plan

### Stage 1: Project Setup and Configuration

1. **Update `crates/py-propag/Cargo.toml`**:
   ```toml
   [package]
   name = "py-propag"
   version = "0.1.0"
   edition = "2021"

   [lib]
   name = "_py_propag"
   crate-type = ["cdylib"]

   [dependencies]
   pyo3 = { version = "0.24", features = ["abi3-py38"] }
   numpy = "0.24"
   propag = { path = "../propag" }
   firelib = { path = "../firelib" }
   geometry = { path = "../geometry" }
   ```

2. **Configure `crates/py-propag/pyproject.toml`**:
   ```toml
   [tool.maturin]
   module-name = "py_propag._py_propag"
   ```

3. **Setup type stubs**:
   - Create `py_propag.pyi` for type annotations
   - Add `py.typed` marker file in the Python package directory

### Stage 2: Core API Design

1. **Wrap C FFI Interface**:
   - Handle error codes and convert to Python exceptions
   - Manage memory allocation/deallocation

2. **Expose Key Data Structures**:
   ```rust
   #[pyclass]
   struct GeoReference { ... }
   
   #[pyclass]
   struct Settings { ... }
   
   #[pyclass]
   struct PropagationResults { ... }
   ```

3. **High-level Python API**:
   ```python
   def propagate(
       ignition_points: List[Tuple[float, float]],
       terrain_file: str,
       settings: Settings
   ) -> PropagationResults:
       """Run wildfire propagation simulation."""
   ```

### Stage 3: Data Handling

1. **NumPy Integration**:
   - Convert terrain grids to/from NumPy arrays
   - Handle fire arrival time matrices
   - Use `numpy` crate for zero-copy operations

2. **GDAL Integration**:
   - Leverage Python's existing GDAL bindings
   - Provide utilities for raster data conversion
   - Support common GIS file formats

### Stage 4: Error Handling and Validation

1. **Input Validation**:
   - Check CUDA device availability at module import
   - Validate terrain data dimensions
   - Verify fuel model parameters against catalog

2. **Error Propagation**:
   - Map CUDA errors to descriptive Python exceptions
   - Handle GDAL errors with context
   - Provide actionable error messages

### Stage 5: Testing and Documentation

1. **Unit Tests**:
   - Test data conversion functions
   - Validate error handling paths
   - Check memory management

2. **Integration Tests**:
   - Full simulation workflow tests
   - Compatibility with QGIS plugin
   - Performance benchmarks

3. **Documentation**:
   - API reference with examples
   - GPU setup guide
   - Installation instructions

## Requirements Specification

### Functional Requirements

1. **Core Simulation API**:
   ```python
   import py_propag
   
   # Configure simulation
   settings = py_propag.Settings(
       max_time_s=3600.0,
       time_step_s=1.0
   )
   
   # Run simulation
   results = py_propag.propagate(
       ignition_points=[(x1, y1), (x2, y2)],
       terrain_file="terrain.tif",
       settings=settings
   )
   
   # Access results
   arrival_times = results.arrival_times  # NumPy array
   ```

2. **Terrain Handling**:
   - Accept GDAL-compatible raster files
   - Support NumPy array input
   - Access fuel model catalog (NFFL 1-13)

### Security Requirements

1. **Input Validation**:
   - Validate file paths (prevent directory traversal)
   - Check array dimensions (prevent buffer overflows)
   - Sanitize fuel model selections

2. **Resource Management**:
   - Limit GPU memory allocation
   - Implement simulation timeouts
   - Clean up CUDA resources on failure

### Performance Requirements

1. **GPU Utilization**:
   - Maintain CUDA kernel performance
   - Minimize Python-Rust data copying
   - Target <100ms Python binding overhead

2. **Memory Efficiency**:
   - Stream large datasets if needed
   - Reuse GPU allocations
   - Profile memory usage

### Style Requirements

1. **Python API**:
   - Follow PEP 8 conventions
   - Use type hints throughout
   - Provide comprehensive docstrings

2. **Error Messages**:
   - Clear and actionable
   - Include GPU requirements
   - Example: "CUDA device not found. This package requires an NVIDIA GPU with CUDA 12.6+"

### Maintainability Requirements

1. **Testing**:
   - Unit tests for all public APIs
   - Mock CUDA for CI/CD
   - GPU integration tests

2. **Documentation**:
   - Sphinx-compatible docstrings
   - README with setup instructions
   - Changelog maintenance

## Milestones

- **M1**: Basic project setup with maturin building successfully
- **M2**: Core FFI wrapper working with simple test case
- **M3**: Full API exposed with NumPy integration
- **M4**: Error handling and validation complete
- **M5**: Tests passing and documentation complete

## Red Herrings (What NOT to Change)

1. **Do NOT modify the core propag crate** - The Python bindings should wrap existing functionality, not alter the core simulation engine
2. **Do NOT implement new fire models** - Use the existing firelib implementation
3. **Do NOT create new CUDA kernels** - Leverage the existing `propag.cu` implementation
4. **Do NOT modify the C FFI interface** - Build on top of `FFIPropagation_run`
5. **Do NOT implement custom GDAL bindings** - Use Python's existing GDAL package

## Developer Guide

### Reference Documentation
- For Rust development practices, reference the Rust style guide if available
- Use sub-agents to analyze the existing propag and firelib crates for API patterns
- Study the QGIS plugin implementation for expected usage patterns

### Development Workflow
1. Use `maturin develop` for rapid iteration
2. Test with real GPU hardware when available
3. Use sub-agents to:
   - Analyze complex Rust structures in propag/firelib
   - Review PyO3 best practices from documentation
   - Examine similar GPU-accelerated Python packages

### Key Files to Reference
- `crates/propag/src/ffi.rs` - C FFI interface
- `crates/propag/src/lib.rs` - Core propag API
- `qgis-plugin/` - Expected Python usage patterns
- `crates/py-propag/` - Existing stub to build upon

### Testing Strategy
1. Start with unit tests for data conversion
2. Add integration tests using sample terrain data
3. Benchmark against C FFI performance
4. Validate compatibility with QGIS plugin

## Implementation Order

1. Set up basic maturin build pipeline
2. Create minimal FFI wrapper
3. Add Settings and GeoReference classes
4. Implement main propagate() function
5. Add NumPy array conversion
6. Implement error handling
7. Write comprehensive tests
8. Create documentation

This plan provides a complete roadmap for implementing Python bindings for the propag crate, leveraging existing infrastructure while addressing the unique challenges of GPU-accelerated geospatial computation.
