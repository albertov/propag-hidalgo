#!/usr/bin/env python3
"""
Basic usage example for py-propag fire simulation library.

This script demonstrates a simple fire propagation simulation on synthetic
terrain data. It shows the complete workflow from terrain setup through
simulation execution and results analysis.
"""

import numpy as np
import sys
import os

# Add the parent directory to Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import py_propag
    print("✓ py-propag imported successfully")
except ImportError as e:
    print(f"✗ Failed to import py-propag: {e}")
    print("Build the module first with: maturin develop --release")
    sys.exit(1)


def check_system_requirements():
    """Check if system meets requirements for fire simulation."""
    print("\n=== System Requirements Check ===")
    
    # Check CUDA availability
    try:
        cuda_available = py_propag.check_cuda_available()
        if cuda_available:
            print("✓ CUDA is available - GPU acceleration enabled")
        else:
            print("⚠ CUDA not detected - simulation will use CPU (slower)")
            print("  Install NVIDIA drivers and CUDA runtime for optimal performance")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False
    
    # Check NumPy
    try:
        print(f"✓ NumPy version: {np.__version__}")
    except:
        print("✗ NumPy not available")
        return False
    
    return True


def create_synthetic_terrain(width, height):
    """Create synthetic terrain data for demonstration."""
    print(f"\n=== Creating Synthetic Terrain ({width}x{height} cells) ===")
    
    # Create coordinate grids
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    
    # Generate elevation with some hills and valleys
    elevation = (
        800 +  # Base elevation (800m)
        200 * np.sin(X / width * 2 * np.pi) * np.cos(Y / height * 2 * np.pi) +  # Rolling hills
        100 * np.random.normal(0, 1, (height, width))  # Random variation
    ).astype(np.float32)
    
    # Ensure elevation is within valid range
    elevation = np.clip(elevation, 500, 1500)
    
    # Generate slope based on elevation gradient
    grad_y, grad_x = np.gradient(elevation)
    slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2))).astype(np.float32)
    slope = np.clip(slope, 0, 45)  # Max 45 degree slope
    
    # Generate aspect from elevation gradient
    aspect = np.degrees(np.arctan2(-grad_x, grad_y)).astype(np.float32)
    aspect = (aspect + 360) % 360  # Convert to 0-360 range
    
    # Create fuel model pattern (mixed forest types)
    fuel_model = np.ones((height, width), dtype=np.uint8) * 2  # Default to fuel model 2
    
    # Add some variety in fuel models
    # Fuel model 1 (short grass) in some areas
    grass_mask = (elevation < 900) & (slope < 10)
    fuel_model[grass_mask] = 1
    
    # Fuel model 4 (chaparral) on steep slopes
    chaparral_mask = slope > 25
    fuel_model[chaparral_mask] = 4
    
    # Fuel model 9 (hardwood litter) in valleys
    valley_mask = elevation < 700
    fuel_model[valley_mask] = 9
    
    # Add some non-fuel areas (water bodies, rock outcrops)
    non_fuel_mask = np.random.random((height, width)) < 0.05  # 5% non-fuel
    fuel_model[non_fuel_mask] = 0
    
    print(f"  - Elevation range: {elevation.min():.1f} - {elevation.max():.1f} m")
    print(f"  - Slope range: {slope.min():.1f} - {slope.max():.1f} degrees")
    print(f"  - Aspect range: {aspect.min():.1f} - {aspect.max():.1f} degrees")
    print(f"  - Fuel models: {np.unique(fuel_model)}")
    
    return elevation, slope, aspect, fuel_model


def setup_spatial_reference(width, height):
    """Set up spatial reference system for the simulation."""
    print("\n=== Setting Up Spatial Reference ===")
    
    # Use UTM Zone 33N (EPSG:32633) - common for Central Europe
    proj_string = "EPSG:32633"
    proj = np.frombuffer(proj_string.encode('utf-8') + b'\x00', dtype=np.uint8)
    
    # Define geo-transform for 30m resolution grid
    # Starting at UTM coordinates (500000, 4000000)
    pixel_size = 30.0  # 30 meter pixels
    transform = np.array([
        500000.0,      # X origin (UTM easting)
        pixel_size,    # Pixel width
        0.0,           # X rotation (0 for north-up)
        4000000.0,     # Y origin (UTM northing)
        0.0,           # Y rotation (0 for north-up)
        -pixel_size    # Pixel height (negative for north-up)
    ], dtype=np.float64)
    
    print(f"  - Coordinate system: {proj_string}")
    print(f"  - Pixel size: {pixel_size}m")
    print(f"  - Grid extent: {width * pixel_size}m x {height * pixel_size}m")
    print(f"  - SW corner: ({transform[0]:.0f}, {transform[3] + transform[5] * height:.0f})")
    print(f"  - NE corner: ({transform[0] + transform[1] * width:.0f}, {transform[3]:.0f})")
    
    return proj, transform


def create_ignition_points(width, height, transform):
    """Create ignition points for the fire simulation."""
    print("\n=== Creating Ignition Points ===")
    
    # Calculate center coordinates in UTM
    center_x = transform[0] + transform[1] * width / 2
    center_y = transform[3] + transform[5] * height / 2
    
    print(f"  - Ignition location: ({center_x:.0f}, {center_y:.0f}) UTM")
    
    # Create WKB point geometry (Well-Known Binary format)
    # Structure: [endianness][type][x][y]
    ignition_wkb = bytearray()
    ignition_wkb.extend([1])  # Little endian
    ignition_wkb.extend([1, 0, 0, 0])  # Point geometry type
    ignition_wkb.extend(np.array(center_x, dtype=np.float64).tobytes())  # X coordinate
    ignition_wkb.extend(np.array(center_y, dtype=np.float64).tobytes())  # Y coordinate
    
    # Create ignition feature starting at time 0
    ignition = py_propag.PyTimeFeature(
        time=0.0,  # Ignition starts at time 0
        geom_wkb=bytes(ignition_wkb)
    )
    
    return [ignition]


def run_simulation(geo_ref, ignition_elements):
    """Run the fire propagation simulation."""
    print("\n=== Running Fire Simulation ===")
    
    # Simulation settings
    max_time = 7200.0  # 2 hours (7200 seconds)
    settings = py_propag.PySettings(geo_ref, max_time)
    
    print(f"  - Maximum simulation time: {max_time/3600:.1f} hours")
    print(f"  - Grid size: {geo_ref.width} x {geo_ref.height} cells")
    print(f"  - Number of ignition points: {len(ignition_elements)}")
    
    # Output paths
    output_path = "fire_simulation_results.tif"
    
    print("  - Starting simulation...")
    
    try:
        # Run the propagation
        results = py_propag.propagate(
            settings=settings,
            output_path=output_path,
            initial_ignited_elements=ignition_elements,
            initial_ignited_elements_crs="EPSG:32633"
        )
        
        print("✓ Simulation completed successfully")
        return results
        
    except py_propag.PropagValidationError as e:
        print(f"✗ Validation error: {e}")
        return None
    except py_propag.PropagCudaError as e:
        print(f"✗ CUDA error: {e}")
        print("  Make sure NVIDIA drivers and CUDA runtime are installed")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def analyze_results(results):
    """Analyze and display simulation results."""
    if results is None:
        print("No results to analyze")
        return
    
    print("\n=== Analyzing Results ===")
    
    # Get arrival times array
    arrival_times = results.arrival_times
    grid_shape = results.shape
    
    print(f"  - Result grid shape: {grid_shape}")
    print(f"  - Arrival times array shape: {arrival_times.shape}")
    
    # Analyze burned area
    burned_mask = np.isfinite(arrival_times)
    burned_cells = np.sum(burned_mask)
    total_cells = arrival_times.size
    burned_fraction = burned_cells / total_cells
    
    # Calculate burned area in hectares (assuming 30m pixels)
    pixel_area_m2 = 30 * 30  # 900 m²
    burned_area_ha = burned_cells * pixel_area_m2 / 10000  # Convert to hectares
    
    print(f"  - Total cells: {total_cells}")
    print(f"  - Burned cells: {burned_cells}")
    print(f"  - Burned fraction: {burned_fraction:.1%}")
    print(f"  - Burned area: {burned_area_ha:.1f} hectares")
    
    if burned_cells > 0:
        # Analyze fire timing
        burned_times = arrival_times[burned_mask]
        print(f"  - First ignition: {burned_times.min():.1f} seconds")
        print(f"  - Last burning: {burned_times.max():.1f} seconds")
        print(f"  - Fire duration: {(burned_times.max() - burned_times.min())/60:.1f} minutes")
        print(f"  - Average arrival time: {burned_times.mean()/60:.1f} minutes")
    
    # Analyze other result components
    boundary_changes = results.boundary_change
    refs_x = results.refs_x
    refs_y = results.refs_y
    
    print(f"  - Boundary change events: {len(boundary_changes)}")
    print(f"  - Reference points: {len(refs_x)}")
    
    # Provide visualization suggestions
    print("\n=== Visualization Suggestions ===")
    print("To visualize results, you can use:")
    print("  1. matplotlib for basic plotting:")
    print("     import matplotlib.pyplot as plt")
    print("     plt.imshow(arrival_times, cmap='hot')")
    print("     plt.colorbar(label='Arrival time (seconds)')")
    print("     plt.show()")
    print()
    print("  2. QGIS or other GIS software to open:")
    print("     fire_simulation_results.tif")


def main():
    """Main function to run the complete fire simulation example."""
    print("py-propag Fire Simulation Example")
    print("=================================")
    
    # Check system requirements
    if not check_system_requirements():
        print("\nSystem requirements not met. Please install required dependencies.")
        return 1
    
    # Simulation parameters
    width, height = 100, 100  # 100x100 grid (3km x 3km at 30m resolution)
    
    try:
        # 1. Create synthetic terrain
        elevation, slope, aspect, fuel_model = create_synthetic_terrain(width, height)
        
        # 2. Set up spatial reference
        proj, transform = setup_spatial_reference(width, height)
        
        # 3. Load terrain data into the simulation
        print("\n=== Loading Terrain Data ===")
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
        print("✓ Terrain data loaded and validated")
        
        # 4. Create ignition points
        ignition_elements = create_ignition_points(width, height, transform)
        
        # 5. Run simulation
        results = run_simulation(geo_ref, ignition_elements)
        
        # 6. Analyze results
        analyze_results(results)
        
        print("\n=== Example Complete ===")
        print("Fire simulation example completed successfully!")
        
        return 0
        
    except py_propag.PropagValidationError as e:
        print(f"\n✗ Validation error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)