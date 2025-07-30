# GPU-Accelerated Solutions to Taxicab Geometry Distortions in Grid-Based Fire Spread Simulation

## Abstract

Building upon the taxicab geometry problems identified by Caballero (2006), this paper presents GPU-accelerated solutions implemented in the propag25 fire spread simulation system. While traditional cellular automata suffer from systematic distance calculation errors when using square-grid maps, leading to distorted fire shapes and inaccurate propagation predictions, modern GPU architectures enable computationally efficient solutions. We describe how parallel CUDA kernels, combined with Euclidean distance calculations and sophisticated boundary tracking, effectively eliminate both the taxicab effect and the knife effect while maintaining real-time performance. The implementation demonstrates that massively parallel computation allows for geometrically accurate fire spread simulation without the computational penalties traditionally associated with corrective algorithms.

**Keywords**: fire spread simulation, CUDA, GPU computing, taxicab geometry, parallel algorithms

## 1. Introduction

Grid-based fire spread simulation has become an essential tool for wildfire management and prediction. However, as Caballero (2006) demonstrated, the discrete nature of square-grid maps introduces systematic errors in distance calculations, leading to distorted fire propagation patterns. These distortions, rooted in taxicab geometry, manifest as kite-shaped fire perimeters instead of the expected elliptical forms and can significantly impact prediction accuracy.

The advent of GPU computing has opened new possibilities for addressing these geometric challenges. Graphics Processing Units (GPUs) excel at parallel computation, making them ideal for cellular automata-based simulations where thousands of cells must be analyzed simultaneously. This paper presents the propag25 system's approach to solving taxicab geometry distortions through GPU-accelerated algorithms.

The key contributions of this work include:
- A CUDA-based implementation that calculates true Euclidean distances in parallel
- An efficient boundary detection system that prevents unrealistic fire propagation
- A reference point tracking mechanism that maintains accurate propagation history
- Performance optimizations that enable real-time simulation without geometric distortions

## 2. Background: The Taxicab Geometry Problem

### 2.1 Extended Taxicab Geometry in Fire Simulation

In traditional cellular automata for fire spread, each cell analyzes its eight neighbors to determine fire propagation. This creates what Caballero termed "Extended Taxicab Geometry," where movement is allowed in eight directions (including diagonals) rather than the four cardinal directions of classic taxicab geometry.

The fundamental problem arises from the accumulation of discrete steps. When fire travels from point A to point B through intermediate cells, the summed distances (L for orthogonal moves, L√2 for diagonal moves) exceed the true Euclidean distance. This discrepancy is most pronounced at angles between the non-distortion axes (multiples of 45°).

### 2.2 Manifestations of Geometric Distortion

Two primary effects emerge from taxicab geometry in fire simulation:

1. **Shape Distortion**: Fire perimeters become octagonal or kite-shaped rather than elliptical, with maximum distortion occurring at 22.5°, 67.5°, 112.5°, etc.

2. **The Knife Effect**: When fire spread ellipses are narrower than cell resolution, propagation along the major axis creates unrealistic "knife-edge" patterns.

## 3. GPU-Based Solutions

### 3.1 Architecture Overview

The propag25 system leverages CUDA's parallel architecture to implement geometrically accurate fire spread simulation. The core propagation kernel processes thousands of cells simultaneously, with each thread responsible for analyzing a single cell's fire propagation.

```cuda
class Propagator {
    // Per-thread propagation analysis
    __device__ void run(unsigned *worked, unsigned *progress,
                       const unsigned int *__restrict__ boundaries);
    __device__ Point find_neighbor_which_reaches_first(
        const unsigned int *__restrict__ boundaries) const;
    __device__ float time_from(const PointRef &from,
                              const FireSimpleCuda &fire) const;
};
```

### 3.2 Iterative Wave Propagation Algorithm

The fire front advances through an iterative wave propagation mechanism. Each iteration does not represent one cell of advancement, but rather a global refinement where all cells simultaneously check if they can be reached faster through any neighboring path. The algorithm continues these refinement iterations until no cell in the entire grid finds a faster arrival time, meaning the fire has either reached all reachable cells within the simulation time limit or has been stopped by barriers:

```cuda
do { // Grid loop - continues until simulation time is reached
    // Phase 1: Load neighbor data into shared memory
    load_points_into_shared_memory();
    
    // Phase 2: Find the neighbor that reaches this cell first
    Point best = find_neighbor_which_reaches_first(boundaries);
    
    // Phase 3: Update cell if a faster path is found
    bool improved = update_point(best);
    
    // Phase 4: Synchronize and check for global progress
    cooperative_groups::this_grid().sync();
    
} while (grid_improved); // Continue while fire is still spreading
```

This iterative approach ensures that fire propagation follows the path of least resistance, with each cell continuously evaluating whether fire from any neighboring cell can reach it faster than its current arrival time. The algorithm naturally handles complex scenarios such as fire flanking around barriers or accelerating through highly combustible fuel types.

### 3.3 Euclidean Distance Calculation

Unlike traditional cellular automata that accumulate grid steps, propag25 calculates true Euclidean distances between any two points:

```cuda
__device__ inline float time_from(const PointRef &from,
                                 const FireSimpleCuda &fire) const {
    float dx = (from_pos.x - to_pos.x) * settings_.geo_ref.transform.gt.dx;
    float dy = (from_pos.y - to_pos.y) * settings_.geo_ref.transform.gt.dy;
    float distance = __fsqrt_rz(dx * dx + dy * dy);
    return from.time + (distance * __frcp_rd(speed));
}
```

This direct calculation eliminates taxicab distortion entirely. The use of CUDA intrinsics (`__fsqrt_rz`, `__frcp_rd`) provides hardware-accelerated computation while maintaining acceptable precision.

### 3.4 Reference Point System and Path Selection

Similar to Caballero's Huygens principle solution, propag25 maintains reference points for each burning cell:

```cuda
struct PointRef {
    float time;
    ushort2 pos;  // Reference position
};

struct Point {
    float time;
    FireSimpleCuda fire;
    PointRef reference;
};
```

This system tracks the origin of fire spread to each cell, enabling accurate back-calculation of propagation times without accumulating discrete step errors. For each neighbor analysis, the algorithm considers two possible propagation paths:

1. **Direct path from neighbor**: The fire travels directly from the neighboring cell
2. **Path through neighbor's reference**: The fire follows the neighbor's historical path to its reference point, then to the current cell

This dual-path consideration is crucial for maintaining physical realism, as fire often follows indirect routes due to wind patterns, terrain features, or fuel distribution.

### 3.5 Boundary Detection and Physical Significance

Boundaries represent locations where fire behavior changes significantly, acting as physical or behavioral barriers to fire propagation. The system implements sophisticated boundary detection to identify these critical transitions:

```cuda
__device__ inline void find_boundaries(unsigned int *boundaries) {
    const Point me = shared_[local_ix_];
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            const Point neighbor = /* load neighbor */;
            if (!similar_fires(neighbor.fire, me.fire) &&
                me.fire.speed_max < neighbor.fire.speed_max) {
                set_boundary(boundaries, global_ix_);
            }
        }
    }
}
```

Boundaries are detected when fire characteristics differ significantly between adjacent cells, representing:

1. **Fuel Type Transitions**: Changes from grassland to forest, or from surface litter to canopy fuels
2. **Natural Barriers**: Rivers, roads, rock outcrops, or other non-combustible features
3. **Fire Intensity Changes**: Transitions between surface fire and crown fire regimes
4. **Meteorological Boundaries**: Sharp changes in wind patterns due to terrain effects

These boundaries are crucial for preventing unrealistic fire behavior, such as fire "teleporting" across rivers or maintaining grassland spread rates through dense forest.

### 3.6 Path Validation with Digital Differential Analyzer

The most sophisticated aspect of the propagation algorithm is its path validation mechanism. When considering propagation from a neighbor's reference point, the system must verify that the path is not blocked by boundaries. This prevents physically impossible scenarios where fire would appear to "jump" across barriers:

```cuda
// Check if the path to neighbor's reference is blocked
DDA iter(neighbor_pos, reference.pos);
while (iter.next(possible_blockage_pos)) {
    if (is_boundary(boundaries, blockage_idx)) {
        // Path is blocked! Must use neighbor as new reference
        reference = neighbor_as_ref;
        candidate_time = neighbor_time;
        break;
    }
}
```

The Digital Differential Analyzer (DDA) traces a line through the grid from the neighbor to its reference point, checking each intervening cell for boundaries. If a boundary is encountered, the algorithm recognizes that the historical path is blocked and reverts to using the neighbor itself as the new reference point.

This mechanism enables realistic fire behavior in complex scenarios:

**Example 1: Fire Flanking a River**
```
Initial state:          After propagation:
A → B → C              A → B ↘
    ↓                       ↓   D
[RIVER] D              [RIVER] ↙
                           E ← F
```
Without path validation, fire at C could incorrectly reach D through A's reference. With validation, the river boundary forces fire to flank around.

**Example 2: Fuel Type Transition**
```
Grassland | Forest | Grassland
Fast → → | Slow | → → Fast
```
Path validation ensures fire must slow down through the forest section rather than maintaining grassland speeds by "skipping" through.

## 4. Performance Optimizations

### 4.1 Shared Memory Usage

Each thread block loads a halo region into shared memory, reducing global memory access:

```cuda
__device__ inline void load_points_into_shared_memory() {
    // Load center point and halo regions
    load_point_at_offset(make_int2(0, 0));
    if (x_near_x0 && y_near_y0) {
        // Load corners
    }
    // ... load edges
    __syncthreads();
}
```

### 4.2 Cooperative Grid Synchronization

The implementation uses CUDA cooperative groups for grid-wide synchronization:

```cuda
cooperative_groups::this_grid().sync();
```

This enables iterative refinement across the entire grid while maintaining consistency.

## 5. Results and Discussion

### 5.1 Geometric Accuracy

By calculating Euclidean distances directly, propag25 eliminates the systematic errors observed in traditional cellular automata. Fire perimeters maintain their elliptical shape regardless of propagation angle, with no distortion at the problematic 22.5° intervals. The iterative wave propagation ensures that fire fronts develop naturally, following physical paths rather than grid artifacts.

### 5.2 Physical Realism

The combination of boundary detection and path validation creates fire behavior that mirrors real-world observations:

- **Irregular Fire Perimeters**: Rather than perfect geometric shapes, fires develop complex perimeters as they encounter and navigate around barriers
- **Flanking Behavior**: Fire naturally flanks around obstacles, creating realistic finger-like projections
- **Fuel-Dependent Spread**: Propagation rates accurately reflect fuel type transitions without artificial acceleration or deceleration
- **Wind-Terrain Interactions**: The reference system captures how wind-driven fires behave differently when encountering terrain features

### 5.3 Computational Efficiency

The parallel nature of GPU computation transforms computationally expensive operations into real-time processes:

- **Simultaneous Cell Analysis**: Thousands of cells evaluate propagation paths in parallel
- **Efficient Boundary Checking**: Bit-packed boundary arrays minimize memory bandwidth
- **Shared Memory Optimization**: Halo regions reduce redundant global memory access
- **Hardware Intrinsics**: CUDA math functions provide rapid distance and angle calculations

The iterative algorithm continues advancing the fire front throughout the entire simulation period. Each iteration represents a refinement wave where cells check if fire from neighboring cells can reach them faster than previously calculated. For a 5-hour simulation on a 4096×4096 grid with 2m resolution, the algorithm may require thousands of iterations as fire propagates across the landscape, potentially covering many kilometers depending on fuel types and weather conditions.

### 5.4 Scalability

The system scales effectively with GPU capabilities, utilizing:
- Thousands of CUDA cores for parallel processing
- High-bandwidth memory for rapid data access
- Cooperative groups for efficient grid-wide synchronization
- Bit-level parallelism for boundary management

## 6. Conclusions

The propag25 system demonstrates that GPU acceleration not only provides computational speed but also enables geometrically accurate and physically realistic solutions to longstanding problems in grid-based fire simulation. The synergy between parallel computation and sophisticated algorithms addresses both the mathematical challenges identified by Caballero and the physical realism required for operational fire prediction.

Key achievements include:

1. **Elimination of Taxicab Distortions**: Direct Euclidean distance calculations remove systematic geometric errors
2. **Physical Path Validation**: DDA-based boundary checking ensures fire follows realistic propagation paths
3. **Adaptive Reference Systems**: Dynamic reference point assignment captures complex fire behavior patterns
4. **Real-time Performance**: Massively parallel processing enables operationally viable simulation speeds
5. **Boundary-Aware Propagation**: Sophisticated detection and handling of fire behavior transitions

The iterative wave propagation algorithm, combined with path validation and boundary detection, creates a system that not only solves theoretical geometric problems but also produces fire behavior that emergency managers can trust. By preventing physically impossible scenarios while maintaining computational efficiency, propag25 bridges the gap between mathematical accuracy and operational utility.

Future work could explore:
- Adaptive grid refinement in areas of rapid fire spread or complex fuel arrangements
- Multi-GPU scaling for continental-scale simulations
- Integration with real-time weather data streams and satellite fire detection
- Machine learning approaches to boundary detection and fuel classification
- Coupling with atmospheric models for smoke dispersion prediction

## 7. Acknowledgments

This work builds upon the foundational research of David Caballero on taxicab geometry in fire spread simulation. The propag25 project is part of the HiDALGO2 initiative, funded by the European Union for high-performance computing applications.

## References

Caballero, D. (2006). Taxicab Geometry: Some problems and solutions for square grid-based fire spread simulation. Forest Ecology and Management, 234(1), S136.

Finney, M. A. (1998). FARSITE: Fire Area Simulator—model development and evaluation. Research Paper RMRS-RP-4. USDA Forest Service.

NVIDIA Corporation. (2023). CUDA C++ Programming Guide. Version 12.3.

Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. Research Paper INT-115. USDA Forest Service.

Tymstra, C., Bryce, R. W., Wotton, B. M., Taylor, S. W., & Armitage, O. B. (2010). Development and structure of Prometheus: the Canadian wildland fire growth simulation model. Information Report NOR-X-417. Natural Resources Canada.