# 3D Point Cloud Magnetic Field Interpolation

A C++ implementation of adaptive magnetic field interpolation using KD-trees and statistical characteristic length estimation for FEA-DEM coupling applications.

## Overview

This project provides efficient algorithms for interpolating magnetic field data from discrete 3D point clouds. It features:

- **KD-Tree Implementation**: Fast nearest neighbor searches in 3D space
- **Statistical Characteristic Length Estimation**: Multiple methods to determine optimal interpolation scales
- **Adaptive Interpolation**: Inverse distance weighting with locally adaptive search radii
- **Magnetic Field Gradient Computation**: Numerical gradient calculation using central differences

## Key Features

- **Efficient Spatial Queries**: O(log n) average time complexity for nearest neighbor searches
- **Robust Scale Estimation**: Combines multiple statistical methods for characteristic length determination
- **Adaptive Search Radii**: Automatically adjusts interpolation neighborhood based on local point density
- **Caching**: Built-in result caching for improved performance on repeated queries
- **OpenMP Support**: Parallel processing for local characteristic length computation

## Dependencies

- C++17 compatible compiler
- CMake 3.10 or higher
- OpenMP (for parallel processing)

## Build Instructions

1. Clone or download the project files
2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
3. Configure with CMake:
   ```bash
   cmake ..
   ```
4. Build the project:
   ```bash
   make
   ```

## Usage

The program includes a comprehensive demo that showcases all major features:

```bash
./point3d_demo
```

### Example Output

```
===========================================
FEA-DEM Magnetic Field Interpolation Characteristic Length Estimation Demo
===========================================
Generating 5000 test points...
Data generation completed

1. Characteristic Length Estimation Test
---------------------
  Nearest neighbor method: 0.587
  Multi-scale method: 0.592
  Density method: 0.589
  Combined global characteristic length: 0.589
  Computation time: 45.2 ms

2. Adaptive Interpolator Test
---------------------
  Query point magnetic field interpolation results:
    Point 1 (5, 5, 5): B = [0.001, 0, 0]
    Point 2 (2, 3, 4): B = [0.0008, -0.0002, 0.0001]
    ...

3. Magnetic Field Gradient Computation Test
---------------------
  Magnetic field gradient at point (5, 5, 5):
    ∇Bx = [0.0001, -0.0001, 0.0002]
    ∇By = [-0.0001, 0.0001, -0.0001]
    ∇Bz = [0.0002, -0.0001, 0.0001]

4. Performance Benchmark
---------------------
  100 interpolation query time: 12.3 ms
  Average time per query: 0.123 ms

===========================================
Demo completed
===========================================
```

## Algorithm Details

### Characteristic Length Estimation

The system uses three complementary methods:

1. **Nearest Neighbor Method**: Average distance to k-nearest neighbors
2. **Multi-scale Method**: Median characteristic length across different k values
3. **Local Density Method**: Characteristic length derived from point density (L ∝ ρ^(-1/3))

### Interpolation

- **Adaptive Search Radius**: Based on local characteristic length
- **Inverse Distance Weighting**: 1/r² weighting with regularization
- **Fallback Strategy**: Expands search radius if insufficient neighbors found

### KD-Tree Implementation

- Balanced binary tree for 3D spatial partitioning
- Efficient nearest neighbor and radius searches
- Automatic tree construction with median splitting

## Performance Characteristics

- **Build Time**: O(n log n) for KD-tree construction
- **Query Time**: O(log n) average for single interpolations
- **Memory Usage**: O(n) for point storage and tree structure
- **Scalability**: Handles point clouds with 10^4 to 10^6 points efficiently

## Applications

- FEA-DEM coupling for magnetic field analysis
- Scientific visualization of magnetic field data
- Electromagnetic simulation preprocessing
- Point cloud data analysis and interpolation

## License

This project is provided as-is for educational and research purposes.