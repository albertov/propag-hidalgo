# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Propag25 is a GPU-accelerated wildfire propagation simulation system built in Rust with CUDA support. The project is part of the HiDALGO2 initiative, funded by the European Union for high-performance computing applications.

## Build and Development Commands

### Prerequisites
- Nix development environment
- NVIDIA GPU with CUDA support (12.6+)
- Rust nightly toolchain (managed via rust-toolchain.toml)

### Core Commands

```bash
 Build the project
cd crates
cargo build

# Run tests
cargo test
# Or use the wrapper script from root:
./run_test.sh

# Format and lint code
./run_fmt.sh
# This runs both `nix fmt` and `cargo clippy --fix`

# Build specific packages
nix run .#make_deb      # Build Debian package
nix run .#make_docker   # Build Docker image
```

### Running Tests for Specific Crates
```bash
cd crates
cargo test -p dmoist    # Test dmoist module
cargo test -p firelib   # Test firelib module
cargo test -p propag    # Test propag module
cargo test -p py-propag # Test Python bindings
```

### Python Bindings Development
```bash
# Build and test Python bindings
cd crates/py-propag
maturin develop --release
python -m pytest tests/

# Run Python examples
python examples/basic_usage.py
```

## Architecture Overview

### Core Components

1. **firelib** (`crates/firelib/`) - Core fire behavior modeling library
   - Implements Rothermel fire spread model. Tests it against a reference
     implementation via firelib-sys FFI
   - Supports both CPU (f32/f64) and CUDA implementations
   - Key modules: `firelib.rs`, `fuel_moisture.rs`, `cuda.rs`
   - Pure Rust implementation with no_std support

2. **dmoist** (`crates/dmoist/`) - Dead fuel moisture calculations
   - Provides fuel moisture calculations
   - Calculates hourly fuel moisture (1hr, 10hr, 100hr timelag classes)
   - Takes meteorological inputs (temperature, humidity, cloud cover)
   - Accounts for terrain factors (slope, aspect) and precipitation history
   - Pure Rust implementation with no_std support

3. **propag** (`crates/propag/`) - GPU-accelerated fire propagation engine
   - CUDA kernel implementation for massively parallel fire spread simulation
   - Manages terrain grid and fire spread calculations
   - Entry point: `main.rs` with GPU initialization and execution

4. **geometry** (`crates/geometry/`) - Geometric utilities for spatial calculations

5. **py-propag** (`crates/py-propag/`) - Python bindings for the propagation engine
   - Provides NumPy-compatible interface for fire simulation
   - Supports CUDA acceleration with error handling
   - Used by QGIS plugin for fire modeling algorithms
   - Comprehensive API for terrain data loading and propagation

6. **QGIS Plugin** (`qgis-plugin/`) - Integration with QGIS GIS software
   - C++ processing provider for QGIS algorithms
   - Python plugin interface leveraging py-propag bindings
   - Geospatial workflow integration for fire modeling

### Key Design Patterns

- **No-std support**: Core libraries (firelib, dmoist) are no_std compatible for embedded/CUDA use
- **FFI Integration**: firelib-sys provides safe Rust bindings to C fireLib implementation
- **GPU Acceleration**: CUDA kernels in propag for parallel fire spread computation
- **Modular Architecture**: Each crate has focused responsibilities with clear interfaces

## Tool Usage Preferences

**IMPORTANT**: When analyzing Rust code in this project, prefer using rust-language-server tools:
- Use `mcp__rust-language-server__diagnostics` to check for compilation errors
- Use `mcp__rust-language-server__hover` to understand types and documentation
- Use `mcp__rust-language-server__definition` to navigate to symbol definitions
- Use `mcp__rust-language-server__references` to find all usages of symbols
- Use `mcp__rust-language-server__rename_symbol` for safe refactoring

These tools provide type-aware analysis that is more accurate than text-based search for Rust code.

## Working with CUDA Code

The project uses Rust-CUDA for GPU programming:
- CUDA kernels are in `crates/propag/src/propag.cu`
- Build configuration in `crates/propag/build.rs`
- Requires NVIDIA Container Toolkit for Docker deployment

## Testing on HPC Systems

The project is designed to run on HPC clusters like Meluxina:
- Uses Apptainer (Singularity) containers
- MPI support for multi-GPU execution
- See README.md for specific Meluxina deployment commands

## Code Quality and Best Practices

### Dependency and Import Guidelines

- **CRITICAL: Avoid Conditional Imports**
  - Never use conditional import patterns that try to handle module availability
  - Bad Example:
    ```python
    # Try to import the module
    try:
        import py_propag
    except ImportError:
        # If the module isn't built yet, mock it for testing
        sys.modules['py_propag'] = MagicMock()
        import py_propag
    ```
  - Good Example:
    ```python
    import py_propag
    ```
  - Rationale: Dependencies are always available. If a dependency is missing, it indicates a build-setup bug that should be fixed, not worked around.
  - **CRITICAL: When writing code, NEVER add conditional imports**
    - Unconditional imports guarantee consistency and clear dependency management
    - If an import fails, it means there's a build or configuration problem that needs immediate resolution
    - Mocking or conditionally importing modules masks underlying setup issues