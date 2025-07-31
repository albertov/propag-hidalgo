"""
Basic functionality tests for py-propag module.

Tests the core functionality including class instantiation, 
property access, and basic operations.
"""

import pytest
import numpy as np
from unittest.mock import patch
import os
import py_propag


class TestCudaAvailability:
    """Test CUDA availability checking functionality."""
    
    def test_check_cuda_available_with_nvidia_smi(self):
        """Test CUDA detection when nvidia-smi is available."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = b'1\n'
            
            result = py_propag.check_cuda_available()
            assert isinstance(result, bool)
    
    def test_check_cuda_available_fallback(self):
        """Test CUDA detection fallback to /dev/nvidia0."""
        with patch('subprocess.run', side_effect=FileNotFoundError), \
             patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = py_propag.check_cuda_available()
            assert isinstance(result, bool)
    
    def test_check_cuda_not_available(self):
        """Test when CUDA is not available."""
        with patch('subprocess.run', side_effect=FileNotFoundError), \
             patch('os.path.exists', return_value=False):
            
            result = py_propag.check_cuda_available()
            assert result is False


class TestPyGeoReference:
    """Test PyGeoReference class functionality."""
    
    def test_geo_reference_creation(self):
        """Test creating a GeoReference object."""
        width, height = 100, 200
        proj = 'EPSG:4326'
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        
        geo_ref = py_propag.PyGeoReference(width, height, proj, transform)
        assert geo_ref.width == width
        assert geo_ref.height == height
    
    def test_geo_reference_invalid_dimensions(self):
        """Test GeoReference with invalid dimensions."""
        proj = 'EPSG:4326'
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        
        with pytest.raises(py_propag.PropagValidationError, match="Grid dimensions must be positive"):
            py_propag.PyGeoReference(0, 100, proj, transform)
    
    def test_geo_reference_invalid_transform(self):
        """Test GeoReference with invalid transform."""
        width, height = 100, 200
        proj = 'EPSG:4326'
        
        # Invalid transform - wrong number of elements
        transform_bad = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        
        with pytest.raises(py_propag.PropagValidationError, match="exactly 6 elements"):
            py_propag.PyGeoReference(width, height, proj, transform_bad)


class TestPySettings:
    """Test PySettings class functionality."""
    
    def test_settings_creation(self):
        """Test creating a Settings object."""
        width, height = 100, 200
        proj = 'EPSG:4326'
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        max_time = 3600.0  # 1 hour
        
        geo_ref = py_propag.PyGeoReference(width, height, proj, transform)
        settings = py_propag.PySettings(geo_ref, max_time)
        assert settings.max_time == max_time
    
    def test_settings_invalid_max_time(self):
        """Test Settings with invalid max_time."""
        width, height = 100, 200
        proj = 'EPSG:4326'
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        
        geo_ref = py_propag.PyGeoReference(width, height, proj, transform)
        
        # Test negative time
        with pytest.raises(py_propag.PropagValidationError, match="must be positive"):
            py_propag.PySettings(geo_ref, -1.0)
        
        # Test infinite time
        with pytest.raises(py_propag.PropagValidationError, match="finite number"):
            py_propag.PySettings(geo_ref, float('inf'))


class TestPyTimeFeature:
    """Test PyTimeFeature class functionality."""
    
    def test_time_feature_creation(self):
        """Test creating a TimeFeature object."""
        time = 300.0  # 5 minutes
        # Simple WKB for a point geometry
        geom_wkb = bytes([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64])
        
        feature = py_propag.PyTimeFeature(time, geom_wkb)
        assert feature.time == time
        assert feature.geom_wkb == geom_wkb
    
    def test_time_feature_invalid_time(self):
        """Test TimeFeature with invalid time."""
        geom_wkb = bytes([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64])
        
        with pytest.raises(py_propag.PropagValidationError, match="non-negative"):
            py_propag.PyTimeFeature(-1.0, geom_wkb)
    
    def test_time_feature_invalid_geometry(self):
        """Test TimeFeature with invalid geometry."""
        time = 300.0
        
        # Empty geometry
        with pytest.raises(py_propag.PropagValidationError, match="cannot be empty"):
            py_propag.PyTimeFeature(time, b'')
        
        # Too short geometry
        with pytest.raises(py_propag.PropagValidationError, match="too short"):
            py_propag.PyTimeFeature(time, b'short')


class TestLoadTerrainData:
    """Test load_terrain_data function."""
    
    def test_load_terrain_data_basic(self):
        """Test loading terrain data with basic parameters."""
        width, height = 10, 10
        proj = 'EPSG:4326'
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        
        # Optional terrain arrays
        elevation = np.random.uniform(0, 1000, (height, width)).astype(np.float32)
        slope = np.random.uniform(0, 45, (height, width)).astype(np.float32)
        aspect = np.random.uniform(0, 360, (height, width)).astype(np.float32)
        fuel_model = np.random.randint(1, 14, (height, width), dtype=np.uint8)
        
        geo_ref = py_propag.load_terrain_data(
            width, height, proj, transform,
            elevation, slope, aspect, fuel_model
        )
        assert hasattr(geo_ref, 'width')
        assert hasattr(geo_ref, 'height')
    
    def test_load_terrain_data_invalid_dimensions(self):
        """Test load_terrain_data with invalid dimensions."""
        proj = 'EPSG:4326'
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        
        with pytest.raises(py_propag.PropagValidationError, match="must be positive"):
            py_propag.load_terrain_data(0, 10, proj, transform, None, None, None, None)


class TestProjectionEdgeCases:
    """Test projection string API edge cases as identified in code review."""
    
    def test_projection_edge_cases(self):
        """Test projection string validation edge cases."""
        width, height = 100, 200
        transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        
        # Test 1: Empty string validation (should raise PropagValidationError)
        with pytest.raises(py_propag.PropagValidationError, match="Projection string cannot be empty"):
            py_propag.PyGeoReference(width, height, "", transform)
        
        # Test 2: Maximum allowed length (1023 characters) - should work
        max_allowed_proj = "EPSG:4326+" + "a" * (1023 - 10)  # 10 chars for "EPSG:4326+"
        assert len(max_allowed_proj) == 1023
        geo_ref_max = py_propag.PyGeoReference(width, height, max_allowed_proj, transform)
        assert geo_ref_max.width == width
        assert geo_ref_max.height == height
        
        # Test 3: String that's too long (1024 characters) - should raise error
        too_long_proj = "EPSG:4326+" + "a" * (1024 - 10)  # 10 chars for "EPSG:4326+"
        assert len(too_long_proj) == 1024
        with pytest.raises(py_propag.PropagValidationError, match="Projection string too large \\(max 1023 characters\\)"):
            py_propag.PyGeoReference(width, height, too_long_proj, transform)
        
        # Test 4: UTF-8 edge cases - strings with special characters
        # Test with Unicode characters (should work as they're valid UTF-8)
        unicode_proj = "EPSG:4326 with émojis 🌍 and spëcial chars: αβγ"
        geo_ref_unicode = py_propag.PyGeoReference(width, height, unicode_proj, transform)
        assert geo_ref_unicode.width == width
        assert geo_ref_unicode.height == height
        
        # Test with multi-byte UTF-8 characters at boundary
        # Create a string that's exactly 1023 bytes with multi-byte chars at the end
        base_proj = "EPSG:4326+"
        remaining_bytes = 1023 - len(base_proj.encode('utf-8'))
        # Use 2-byte UTF-8 character (é) to test boundary conditions
        num_multibyte_chars = remaining_bytes // 2
        multibyte_proj = base_proj + "é" * num_multibyte_chars
        
        # Adjust to exactly 1023 bytes if needed
        while len(multibyte_proj.encode('utf-8')) > 1023:
            multibyte_proj = multibyte_proj[:-1]
        while len(multibyte_proj.encode('utf-8')) < 1023:
            multibyte_proj += "a"
        
        assert len(multibyte_proj.encode('utf-8')) == 1023
        geo_ref_multibyte = py_propag.PyGeoReference(width, height, multibyte_proj, transform)
        assert geo_ref_multibyte.width == width
        assert geo_ref_multibyte.height == height
        
        # Test string that becomes too long in UTF-8 encoding
        # Use 3-byte UTF-8 characters (like €) to create a string that exceeds byte limit
        base_proj_utf8 = "EPSG:4326+"
        remaining_bytes_utf8 = 1024 - len(base_proj_utf8.encode('utf-8'))
        # Use 3-byte UTF-8 character (€) - this will make the string exceed 1023 bytes
        utf8_long_proj = base_proj_utf8 + "€" * (remaining_bytes_utf8 // 3 + 1)
        
        # Ensure it's over the byte limit
        while len(utf8_long_proj.encode('utf-8')) <= 1023:
            utf8_long_proj += "€"
        
        assert len(utf8_long_proj.encode('utf-8')) >= 1024
        with pytest.raises(py_propag.PropagValidationError, match="Projection string too large \\(max 1023 characters\\)"):
            py_propag.PyGeoReference(width, height, utf8_long_proj, transform)


if __name__ == '__main__':
    pytest.main([__file__])
