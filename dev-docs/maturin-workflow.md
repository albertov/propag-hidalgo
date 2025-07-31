# Maturin Development Workflow Guide

## Overview

Maturin is a tool for building and publishing Python extensions written in Rust using PyO3, providing a seamless bridge between Rust and Python. This guide covers the complete development workflow for the py-propag project and general maturin best practices.

## Prerequisites

### System Requirements
- Python 3.8+ (our project requires >=3.8)
- Rust toolchain (nightly for CUDA support)
- Git for version control
- Virtual environment support

### Installing Maturin
```bash
pip install -U pip maturin
```

## Project Structure

Our py-propag project follows this structure:
```
crates/py-propag/
├── Cargo.toml          # Rust package configuration
├── pyproject.toml      # Python package metadata
├── src/
│   └── lib.rs          # Rust source with PyO3 bindings
├── tests/
│   ├── test_basic.py   # Basic functionality tests
│   ├── test_numpy.py   # NumPy integration tests
│   └── test_validation.py
├── examples/
│   └── basic_usage.py  # Usage examples
├── py.typed            # Type stub marker
└── py_propag.pyi       # Type definitions
```

## Setting Up Development Environment

### 1. Create and Activate Virtual Environment
```bash
# Navigate to py-propag directory
cd /var/lib/mcp-proxy/workspace/crates/py-propag

# Create virtual environment
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install Development Dependencies
```bash
# Install maturin and testing tools
pip install -U pip maturin pytest numpy

# Install project dependencies if any
pip install -r requirements.txt  # if exists
```

## Core Development Commands

### Building for Development

#### `maturin develop`
**Primary development command** - builds and installs the extension directly into the virtual environment:

```bash
# Debug build (fast compilation, slower runtime)
maturin develop

# Release build (slower compilation, optimized runtime)
maturin develop --release

# With specific features
maturin develop --features pyo3/extension-module
```

**Key characteristics:**
- Fastest iteration cycle for development
- Installs directly in current virtualenv
- Skips wheel generation
- Supports editable installs for mixed Rust/Python projects

#### `maturin build`
**Production build command** - creates distributable wheels:

```bash
# Build wheel in target/wheels/
maturin build

# Release build
maturin build --release

# Build for multiple Python versions
maturin build --interpreter python3.8 python3.9 python3.10
```

### Development Workflow

#### Typical Development Cycle
```bash
# 1. Make changes to Rust code in src/lib.rs
# 2. Rebuild and install
maturin develop

# 3. Test changes
python -m pytest tests/

# 4. Run specific test file
python -m pytest tests/test_basic.py -v

# 5. Interactive testing
python -c "import py_propag; print(py_propag.check_cuda_available())"
```

#### Fast Iteration for Mixed Projects
For projects with both Rust and Python code:
```bash
# Use editable install
pip install -e .

# Only rebuild when Rust code changes
maturin develop  # Only needed after Rust changes

# Python-only changes are picked up automatically
python -m pytest tests/
```

## Testing Strategies

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_basic.py

# Run specific test
python -m pytest tests/test_basic.py::TestCudaAvailability::test_check_cuda_available_with_nvidia_smi
```

#### Testing with Mock Support
Our tests include intelligent mocking for when the module isn't built:
```python
# Tests handle missing module gracefully
try:
    import py_propag
except ImportError:
    sys.modules['py_propag'] = MagicMock()
    import py_propag
```

#### Rust Unit Tests
```bash
# Test Rust code without Python bindings
cd /var/lib/mcp-proxy/workspace/crates/py-propag
cargo test --no-default-features

# Test with all features
cargo test
```

### Continuous Integration Testing
```bash
# Generate CI configuration
maturin generate-ci github > .github/workflows/CI.yml

# Test across Python versions
maturin build --interpreter python3.8 python3.9 python3.10 python3.11
```

## Virtual Environment Best Practices

### Environment Management
```bash
# Always use virtual environments
python3 -m venv .venv
source .venv/bin/activate

# Verify environment
which python
which pip

# Deactivate when done
deactivate
```

### Dependency Management
```bash
# Pin maturin version for reproducibility
echo "maturin>=1.7,<2.0" > requirements-dev.txt

# Install dev dependencies
pip install -r requirements-dev.txt

# Generate requirements from current environment
pip freeze > requirements.txt
```

## IDE Integration

### VSCode Configuration
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "rust-analyzer.cargo.features": ["pyo3/extension-module"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

### PyCharm Configuration
1. Set Python interpreter to `.venv/bin/python`
2. Enable Rust plugin for syntax highlighting
3. Configure pytest as test runner
4. Set working directory to project root

### Debug Configuration
For debugging Python-Rust integration:
```bash
# Build with debug symbols
maturin develop --release=false

# Use Python debugger normally
python -m pdb your_script.py
```

## Common Issues and Solutions

### Virtual Environment Issues

**Problem**: "Expected python to be a python interpreter inside a virtualenv"
```bash
# Solutions:
1. Reinstall maturin: pip uninstall maturin && pip install maturin
2. Ensure virtual environment is activated
3. Check environment variables: echo $VIRTUAL_ENV
4. Use explicit interpreter: maturin develop --interpreter .venv/bin/python
```

**Problem**: Permission denied in WSL2
```bash
# Solution:
chmod +x .venv/bin/python
# Or create new virtual environment
```

### Compilation Issues

**Problem**: "Cargo metadata failed"
```bash
# Solutions:
1. Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
2. Update Rust: rustup update
3. Check Cargo: cargo --version
```

**Problem**: Missing dependencies
```bash
# For our project specifically:
cd /var/lib/mcp-proxy/workspace/crates
cargo check  # Verify all dependencies
maturin develop  # From py-propag directory
```

### Module Import Issues

**Problem**: Module not found after build
```bash
# Solutions:
1. Verify installation: pip list | grep py-propag
2. Check Python path: python -c "import sys; print(sys.path)"
3. Reinstall: maturin develop --force
```

### Performance Issues

**Problem**: Slow compilation
```bash
# Solutions:
1. Use debug builds during development: maturin develop
2. Enable incremental compilation in Cargo.toml:
[profile.dev]
incremental = true
```

## CI/CD Considerations

### GitHub Actions Workflow
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install Rust
      uses: dtolnay/rust-toolchain@stable
    - name: Install maturin
      run: pip install maturin[patchelf]
    - name: Build and test
      run: |
        cd crates/py-propag
        maturin develop
        python -m pytest
```

### Building for Distribution
```bash
# Build wheels for multiple platforms
maturin build --release --strip

# Build for specific architectures
maturin build --target x86_64-unknown-linux-gnu

# Upload to PyPI (when ready)
maturin publish --username __token__ --password $PYPI_TOKEN
```

## Project-Specific Commands

### For py-propag Development
```bash
# Navigate to project
cd /var/lib/mcp-proxy/workspace/crates/py-propag

# Setup development environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pytest numpy

# Build and test
maturin develop
python -m pytest tests/ -v

# Run example
python examples/basic_usage.py

# Check CUDA availability
python -c "import py_propag; print('CUDA available:', py_propag.check_cuda_available())"

# Test specific functionality
python -c "
import py_propag
import numpy as np
geo_ref = py_propag.load_terrain_data(
    10, 10,
    np.array(b'EPSG:4326\x00', dtype=np.uint8),
    np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
)
print('Terrain data loaded successfully')
"
```

### Integration with Parent Project
```bash
# Build all crates from parent directory
cd /var/lib/mcp-proxy/workspace/crates
cargo build

# Test firelib integration
cd py-propag
python -c "
import py_propag
print('Available functions:', [f for f in dir(py_propag) if not f.startswith('_')])
"
```

## Performance Optimization

### Build Optimization
```bash
# Release builds for performance testing
maturin develop --release

# Profile-guided optimization (advanced)
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" maturin develop --release
# Run representative workload
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data" maturin develop --release
```

### Memory Management
```bash
# Monitor memory usage during development
python -c "
import py_propag
import tracemalloc
tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.1f} MB')
"
```

## Conclusion

This workflow provides a comprehensive approach to developing Python extensions with maturin, covering everything from initial setup to production deployment. The key to successful maturin development is:

1. **Proper environment setup** with virtual environments
2. **Understanding the difference** between `maturin develop` and `maturin build`
3. **Comprehensive testing** with both Python and Rust test suites
4. **IDE integration** for efficient development
5. **Proactive troubleshooting** of common issues

For the py-propag project specifically, this workflow enables efficient development of the GPU-accelerated wildfire propagation simulation system while maintaining code quality and performance.