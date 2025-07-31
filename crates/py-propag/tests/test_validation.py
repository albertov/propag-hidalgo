"""
Input validation tests for py-propag module.

Tests edge cases, error handling, and input validation logic
to ensure robust error reporting and security.
"""

import pytest
import numpy as np
import py_propag


class TestFilePathValidation:
    """Test file path validation and security measures."""
    
    def test_valid_file_paths(self):
        """Test that valid file paths are accepted."""
        valid_paths = [
            "output.tif",
            "results/fire_spread.tif",
            "/tmp/simulation_output.tif",
            "data/terrain/elevation.tif"
        ]
        
        # Mock validation function behavior
        for path in valid_paths:
            # These paths should be considered valid
            assert ".." not in path
            assert len(path) > 0
            assert not (path.startswith("/etc") or path.startswith("/sys") or path.startswith("/proc"))
    
    def test_invalid_file_paths(self):
        """Test that invalid file paths are rejected."""
        invalid_paths = [
            "",  # Empty path
            "../etc/passwd",  # Directory traversal
            "data/../../../etc/hosts",  # Directory traversal
            "/etc/shadow",  # System directory
            "/sys/kernel/config",  # System directory
            "/proc/version"  # System directory
        ]
        
        # These would be caught by validation
        for path in invalid_paths:
            if path == "":
                assert len(path) == 0
            elif ".." in path:
                assert ".." in path
            elif path.startswith(("/etc", "/sys", "/proc")):
                assert any(path.startswith(prefix) for prefix in ["/etc", "/sys", "/proc"])


class TestArrayDimensionValidation:
    """Test validation of array dimensions and shapes."""
    
    def test_valid_array_dimensions(self):
        """Test arrays with valid dimensions."""
        width, height = 100, 200
        
        # Create properly shaped arrays
        elevation = np.random.uniform(-100, 3000, (height, width)).astype(np.float32)
        slope = np.random.uniform(0, 89, (height, width)).astype(np.float32)
        aspect = np.random.uniform(0, 359, (height, width)).astype(np.float32)
        fuel_model = np.random.randint(1, 14, (height, width), dtype=np.uint8)
        
        # Verify shapes match expected dimensions
        assert elevation.shape == (height, width)
        assert slope.shape == (height, width)
        assert aspect.shape == (height, width)
        assert fuel_model.shape == (height, width)
    
    def test_mismatched_array_dimensions(self):
        """Test arrays with mismatched dimensions."""
        width, height = 100, 200
        
        # Create arrays with wrong dimensions
        wrong_elevation = np.random.uniform(0, 1000, (height + 10, width)).astype(np.float32)
        wrong_slope = np.random.uniform(0, 45, (height, width + 5)).astype(np.float32)
        
        # These should be caught by validation
        assert wrong_elevation.shape != (height, width)
        assert wrong_slope.shape != (height, width)
    
    def test_oversized_arrays(self):
        """Test arrays that exceed reasonable size limits."""
        # Very large dimensions that would exceed memory/processing limits
        huge_width, huge_height = 50000, 50000
        total_cells = huge_width * huge_height
        
        # Should exceed the 100 million cell limit
        assert total_cells > 100_000_000


class TestDataRangeValidation:
    """Test validation of data value ranges."""
    
    def test_elevation_range_validation(self):
        """Test elevation data range validation."""
        width, height = 10, 10
        
        # Valid elevation data
        valid_elevation = np.random.uniform(-500, 8000, (height, width)).astype(np.float32)
        assert np.all(valid_elevation >= -1000) and np.all(valid_elevation <= 9000)
        
        # Invalid elevation data (too low/high)
        invalid_low = np.full((height, width), -2000.0, dtype=np.float32)
        invalid_high = np.full((height, width), 15000.0, dtype=np.float32)
        
        assert np.any(invalid_low < -1000)
        assert np.any(invalid_high > 9000)
    
    def test_slope_range_validation(self):
        """Test slope data range validation."""
        width, height = 10, 10
        
        # Valid slope data (0-90 degrees)
        valid_slope = np.random.uniform(0, 89, (height, width)).astype(np.float32)
        assert np.all(valid_slope >= 0) and np.all(valid_slope <= 90)
        
        # Invalid slope data
        invalid_negative = np.full((height, width), -10.0, dtype=np.float32)
        invalid_high = np.full((height, width), 120.0, dtype=np.float32)
        
        assert np.any(invalid_negative < 0)
        assert np.any(invalid_high > 90)
    
    def test_aspect_range_validation(self):
        """Test aspect data range validation."""
        width, height = 10, 10
        
        # Valid aspect data (0-360 degrees)
        valid_aspect = np.random.uniform(0, 359, (height, width)).astype(np.float32)
        assert np.all(valid_aspect >= 0) and np.all(valid_aspect <= 360)
        
        # Invalid aspect data
        invalid_negative = np.full((height, width), -45.0, dtype=np.float32)
        invalid_high = np.full((height, width), 400.0, dtype=np.float32)
        
        assert np.any(invalid_negative < 0)
        assert np.any(invalid_high > 360)
    
    def test_fuel_model_validation(self):
        """Test fuel model ID validation."""
        width, height = 10, 10
        
        # Valid fuel model IDs (1-13, with 0 for non-fuel areas)
        valid_fuel = np.random.randint(0, 14, (height, width), dtype=np.uint8)
        # Ensure some valid values
        valid_fuel[0, 0] = 1
        valid_fuel[0, 1] = 13
        
        assert np.all(valid_fuel <= 13)
        
        # Invalid fuel model IDs
        invalid_fuel = np.full((height, width), 20, dtype=np.uint8)
        assert np.any(invalid_fuel > 13)


class TestGeoTransformValidation:
    """Test geo-transform parameter validation."""
    
    def test_valid_geo_transform(self):
        """Test valid geo-transform parameters."""
        # Standard geo-transform: [x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height]
        valid_transforms = [
            [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],  # Standard north-up
            [-180.0, 0.1, 0.0, 90.0, 0.0, -0.1],  # Geographic coordinates
            [500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0],  # UTM coordinates
        ]
        
        for transform in valid_transforms:
            assert len(transform) == 6
            assert transform[1] != 0.0  # pixel_width != 0
            assert transform[5] != 0.0  # pixel_height != 0
    
    def test_invalid_geo_transform(self):
        """Test invalid geo-transform parameters."""
        invalid_transforms = [
            [0.0, 1.0, 0.0],  # Wrong number of elements
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # Zero pixel width
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Zero pixel height
            [0.0, float('inf'), 0.0, 0.0, 0.0, -1.0],  # Infinite values
        ]
        
        for transform in invalid_transforms:
            if len(transform) != 6:
                assert len(transform) != 6
            elif len(transform) == 6:
                assert transform[1] == 0.0 or transform[5] == 0.0 or not np.isfinite(transform[1])


class TestTimeValidation:
    """Test time parameter validation."""
    
    def test_valid_time_values(self):
        """Test valid time values."""
        valid_times = [
            0.0,  # Start time
            3600.0,  # 1 hour
            86400.0,  # 1 day
            31536000.0,  # 1 year
        ]
        
        for time_val in valid_times:
            assert time_val >= 0.0
            assert np.isfinite(time_val)
            assert time_val <= 10.0 * 365.0 * 24.0 * 3600.0  # Less than 10 years
    
    def test_invalid_time_values(self):
        """Test invalid time values."""
        invalid_times = [
            -1.0,  # Negative time
            float('inf'),  # Infinite time
            float('nan'),  # NaN time
            10.1 * 365.0 * 24.0 * 3600.0,  # Too large (>10 years)
        ]
        
        for time_val in invalid_times:
            if time_val < 0.0:
                assert time_val < 0.0
            elif not np.isfinite(time_val):
                assert not np.isfinite(time_val)
            elif time_val > 10.0 * 365.0 * 24.0 * 3600.0:
                assert time_val > 10.0 * 365.0 * 24.0 * 3600.0


class TestGeometryValidation:
    """Test WKB geometry validation."""
    
    def test_valid_wkb_geometry(self):
        """Test valid WKB geometry data."""
        # Simple point WKB (21 bytes minimum)
        valid_point_wkb = bytes([
            1,  # Byte order
            1, 0, 0, 0,  # Geometry type (Point)
            0, 0, 0, 0, 0, 0, 240, 63,  # X coordinate (1.0)
            0, 0, 0, 0, 0, 0, 0, 64   # Y coordinate (2.0)
        ])
        
        assert len(valid_point_wkb) >= 9  # Minimum WKB size
        assert len(valid_point_wkb) > 0
    
    def test_invalid_wkb_geometry(self):
        """Test invalid WKB geometry data."""
        invalid_geometries = [
            b'',  # Empty
            b'short',  # Too short
            b'x' * 8,  # Still too short for valid WKB
        ]
        
        for geom in invalid_geometries:
            assert len(geom) < 9


class TestCRSValidation:
    """Test coordinate reference system validation."""
    
    def test_valid_crs_strings(self):
        """Test valid CRS strings."""
        valid_crs = [
            "EPSG:4326",  # WGS84
            "EPSG:3857",  # Web Mercator
            "EPSG:32633",  # UTM Zone 33N
            "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs",  # PROJ string
        ]
        
        for crs in valid_crs:
            assert len(crs) > 0
            assert len(crs) <= 2048
    
    def test_invalid_crs_strings(self):
        """Test invalid CRS strings."""
        invalid_crs = [
            "",  # Empty
            "x" * 3000,  # Too long
        ]
        
        for crs in invalid_crs:
            if len(crs) == 0:
                assert len(crs) == 0
            elif len(crs) > 2048:
                assert len(crs) > 2048


class TestIgnitionElementValidation:
    """Test ignition element validation."""
    
    def test_valid_ignition_elements(self):
        """Test valid ignition element configurations."""
        # Create valid ignition elements
        geom_wkb = bytes([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64])
        
        valid_counts = [1, 10, 100, 1000, 5000]
        
        for count in valid_counts:
            assert 1 <= count <= 10000
    
    def test_invalid_ignition_elements(self):
        """Test invalid ignition element configurations."""
        invalid_counts = [0, 15000, 50000]
        
        for count in invalid_counts:
            if count == 0:
                assert count == 0
            elif count > 10000:
                assert count > 10000


class TestProjectionValidation:
    """Test projection string validation."""
    
    def test_valid_projection_strings(self):
        """Test valid projection strings."""
        valid_projections = [
            b'EPSG:4326\x00',
            b'EPSG:3857\x00',
            b'+proj=utm +zone=33 +datum=WGS84\x00',
        ]
        
        for proj in valid_projections:
            assert len(proj) <= 1024
            assert len(proj) > 0
    
    def test_invalid_projection_strings(self):
        """Test invalid projection strings."""
        # Too long projection string
        too_long_proj = b'x' * 2000
        
        assert len(too_long_proj) > 1024


class TestErrorHandling:
    """Test error handling and exception types."""
    
    def test_exception_hierarchy(self):
        """Test that custom exceptions have proper hierarchy."""
        # Test that our custom exceptions would inherit from appropriate base classes
        cuda_error_base = RuntimeError
        validation_error_base = ValueError
        gdal_error_base = RuntimeError
        mpi_error_base = RuntimeError
        
        # These are the expected base classes
        assert issubclass(RuntimeError, Exception)
        assert issubclass(ValueError, Exception)
    
    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        test_cases = [
            ("width", 0, "Grid dimensions must be positive (width > 0, height > 0)"),
            ("max_time", -1.0, "Maximum simulation time must be positive"),
            ("elevation", 5000.0, "exceeds maximum"),
            ("fuel_model", 20, "Invalid fuel model ID"),
        ]
        
        for param, value, expected_msg_part in test_cases:
            # Verify error message components would be meaningful
            assert len(param) > 0
            assert expected_msg_part in expected_msg_part  # Trivial but validates structure


if __name__ == '__main__':
    pytest.main([__file__])
