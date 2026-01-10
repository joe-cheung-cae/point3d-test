#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <queue>
#include <random>
#include <chrono>
#include <unordered_map>
#include <functional>

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reverse.h>

// ============================================
// Basic Data Structures
// ============================================
// This section defines fundamental data structures used throughout the application,
// including 3D points, hash functions, and field data representations.

struct Point3D {
    double x, y, z;
    
    Point3D() : x(0), y(0), z(0) {}
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    double distance(const Point3D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    
    double squaredDistance(const Point3D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return dx*dx + dy*dy + dz*dz;
    }
    
    // Custom equality operator for hash mapping
    bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Custom hash function for Point3D to enable use in unordered containers
struct Point3DHash {
    std::size_t operator()(const Point3D& p) const {
        std::size_t h1 = std::hash<double>{}(p.x);
        std::size_t h2 = std::hash<double>{}(p.y);
        std::size_t h3 = std::hash<double>{}(p.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Structure to store magnetic field data at specific 3D positions
struct FieldData {
    Point3D position;
    std::array<double, 3> B_field;  // Bx, By, Bz
};

// ============================================
// KD-Tree Implementation (for Fast Nearest Neighbor Search)
// ============================================
// This section implements a KD-tree data structure for efficient spatial queries.
// KD-trees enable O(log n) average time complexity for nearest neighbor searches
// in 3D space, which is crucial for interpolation algorithms.

class KDTree {
private:
    struct Node {
        Point3D point;
        int index;
        int axis;  // 0:x, 1:y, 2:z
        Node* left;
        Node* right;
        
        Node(const Point3D& p, int idx, int ax) 
            : point(p), index(idx), axis(ax), left(nullptr), right(nullptr) {}
    };
    
    Node* root_;
    thrust::host_vector<Point3D> points_;

    Node* buildTree(thrust::host_vector<std::pair<Point3D, int>>& points_vec,
                   int start, int end, int depth) {
        if (start >= end) return nullptr;
        
        int axis = depth % 3;
        
        // Sort points along the current axis (x, y, or z) for balanced tree construction
        thrust::sort(points_vec.begin() + start, points_vec.begin() + end,
                 [axis](const auto& a, const auto& b) {
                     if (axis == 0) return a.first.x < b.first.x;
                     if (axis == 1) return a.first.y < b.first.y;
                     return a.first.z < b.first.z;
                 });
        
        int mid = start + (end - start) / 2;
        Node* node = new Node(points_vec[mid].first, points_vec[mid].second, axis);
        
        node->left = buildTree(points_vec, start, mid, depth + 1);
        node->right = buildTree(points_vec, mid + 1, end, depth + 1);
        
        return node;
    }
    
    void kNearestNeighbors(Node* node, const Point3D& query,
                          std::priority_queue<std::pair<double, int>>& heap,
                          int k) const {
        if (!node) return;
        
        double dist_sq = query.squaredDistance(node->point);
        heap.push({dist_sq, node->index});
        
        if (heap.size() > k) {
            heap.pop();
        }
        
        double diff;
        if (node->axis == 0) {
            diff = query.x - node->point.x;
        } else if (node->axis == 1) {
            diff = query.y - node->point.y;
        } else {
            diff = query.z - node->point.z;
        }
        
        Node* first = diff <= 0 ? node->left : node->right;
        Node* second = diff <= 0 ? node->right : node->left;
        
        kNearestNeighbors(first, query, heap, k);
        
        // Check if the other subtree could contain closer points based on the splitting plane
        if (heap.size() < k || diff * diff < heap.top().first) {
            kNearestNeighbors(second, query, heap, k);
        }
    }
    
    void radiusSearch(Node* node, const Point3D& query, double radius_sq,
                     thrust::host_vector<int>& results) const {
        if (!node) return;
        
        double dist_sq = query.squaredDistance(node->point);
        if (dist_sq <= radius_sq) {
            results.push_back(node->index);
        }
        
        double diff;
        if (node->axis == 0) {
            diff = query.x - node->point.x;
        } else if (node->axis == 1) {
            diff = query.y - node->point.y;
        } else {
            diff = query.z - node->point.z;
        }
        
        if (diff <= 0) {
            radiusSearch(node->left, query, radius_sq, results);
            if (diff * diff <= radius_sq) {
                radiusSearch(node->right, query, radius_sq, results);
            }
        } else {
            radiusSearch(node->right, query, radius_sq, results);
            if (diff * diff <= radius_sq) {
                radiusSearch(node->left, query, radius_sq, results);
            }
        }
    }
    
public:
    KDTree(const thrust::host_vector<Point3D>& points) : points_(points) {
        thrust::host_vector<std::pair<Point3D, int>> points_vec;
        points_vec.reserve(points.size());

        for (size_t i = 0; i < points.size(); ++i) {
            points_vec.push_back(std::make_pair(points[i], i));
        }

        root_ = buildTree(points_vec, 0, points.size(), 0);
    }
    
    ~KDTree() {
        // Recursively delete all nodes to prevent memory leaks
        std::function<void(Node*)> deleteNode = [&](Node* node) {
            if (!node) return;
            deleteNode(node->left);
            deleteNode(node->right);
            delete node;
        };
        deleteNode(root_);
    }
    
    // Find k nearest neighbors to the query point
    thrust::host_vector<std::pair<int, double>> kNearestNeighbors(const Point3D& query, int k) const {
        std::priority_queue<std::pair<double, int>> heap;

        kNearestNeighbors(root_, query, heap, k);

        thrust::host_vector<std::pair<int, double>> results;
        while (!heap.empty()) {
            results.push_back(std::make_pair(heap.top().second, heap.top().first));
            heap.pop();
        }
        thrust::reverse(results.begin(), results.end()); // Convert to distances from small to large

        return results;
    }
    
    // Find all points within a given radius of the query point
    thrust::host_vector<int> radiusSearch(const Point3D& query, double radius) const {
        thrust::host_vector<int> results;
        double radius_sq = radius * radius;
        radiusSearch(root_, query, radius_sq, results);
        return results;
    }
    
    // Find the single nearest neighbor to the query point
    std::pair<int, double> nearestNeighbor(const Point3D& query) const {
        auto neighbors = kNearestNeighbors(query, 1);
        if (neighbors.empty()) return {-1, 0.0};
        return neighbors[0];
    }
};

// ============================================
// Statistical Characteristic Length Estimator Based on Point Clouds
// ============================================
// This class provides multiple methods to estimate the characteristic length scale
// of a point cloud, which is essential for adaptive interpolation algorithms.
// Characteristic length represents the typical distance between neighboring points.

class StatisticalLengthEstimator {
private:
    thrust::host_vector<Point3D> points_;
    std::unique_ptr<KDTree> kdtree_;

public:
    StatisticalLengthEstimator(const thrust::host_vector<Point3D>& points)
        : points_(points) {
        kdtree_ = std::make_unique<KDTree>(points_);
    }
    
    // Method 1: Simple estimation based on nearest neighbor distances
    // This method samples random points and computes the average distance to their nearest neighbors
    double estimateFromNearestNeighbors(int sample_count = 1000) {
        if (points_.empty()) return 0.0;
        
        double sum_distances = 0.0;
        int actual_samples = std::min(sample_count, static_cast<int>(points_.size()));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points_.size() - 1);
        
        for (int i = 0; i < actual_samples; ++i) {
            int idx = dis(gen);
            auto neighbors = kdtree_->kNearestNeighbors(points_[idx], 2); // Self + nearest neighbor
            if (neighbors.size() > 1) {
                sum_distances += std::sqrt(neighbors[1].second);
            }
        }
        
        return sum_distances / actual_samples;
    }
    
    // Method 2: Estimation based on local density
    // Estimates characteristic length using the relationship L ∝ ρ^(-1/3) where ρ is local density
    double estimateFromLocalDensity(double search_radius) {
        if (points_.empty()) return 0.0;
        
        double total_density = 0.0;
        int sample_count = std::min(500, static_cast<int>(points_.size()));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points_.size() - 1);
        
        for (int i = 0; i < sample_count; ++i) {
            int idx = dis(gen);
            auto neighbors = kdtree_->radiusSearch(points_[idx], search_radius);
            
            // Exclude self
            int neighbor_count = 0;
            for (int n_idx : neighbors) {
                if (n_idx != idx) neighbor_count++;
            }
            
            double volume = (4.0/3.0) * M_PI * std::pow(search_radius, 3);
            if (volume > 0) {
                total_density += neighbor_count / volume;
            }
        }
        
        double avg_density = total_density / sample_count;
        if (avg_density <= 0) return 0.0;
        
        // Characteristic length ∝ density^(-1/3)
        return std::pow(1.0 / avg_density, 1.0/3.0);
    }
    
    // Method 3: Multi-scale statistical estimation (most robust)
    // Uses multiple k-values to compute characteristic lengths at different scales and takes the median
    double estimateMultiScale(int max_k = 50) {
        if (points_.empty()) return 0.0;

        thrust::host_vector<double> characteristic_lengths;

        // Calculate characteristic length at different scales
        for (int k = 2; k <= max_k; k *= 2) {
            double sum_kdist = 0.0;
            int valid_samples = 0;

            int sample_count = std::min(200, static_cast<int>(points_.size()));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, points_.size() - 1);

            for (int i = 0; i < sample_count; ++i) {
                int idx = dis(gen);
                auto neighbors = kdtree_->kNearestNeighbors(points_[idx], k + 1);

                if (neighbors.size() > k) {
                    // Distance to the k-th nearest neighbor
                    sum_kdist += std::sqrt(neighbors[k].second);
                    valid_samples++;
                }
            }

            if (valid_samples > 0) {
                characteristic_lengths.push_back(sum_kdist / valid_samples);
            }
        }

        if (characteristic_lengths.empty()) return 0.0;

        // Take median as final estimate (more robust to outliers)
        thrust::sort(characteristic_lengths.begin(), characteristic_lengths.end());
        return characteristic_lengths[characteristic_lengths.size() / 2];
    }
    
    // Method 4: Compute local characteristic lengths for each point
    // Calculates individual characteristic lengths based on k-nearest neighbors for each point
    thrust::host_vector<double> computeLocalCharacteristicLengths(int k_neighbors = 6) {
        thrust::host_vector<double> local_lengths(points_.size(), 0.0);

        #pragma omp parallel for
        for (size_t i = 0; i < points_.size(); ++i) {
            auto neighbors = kdtree_->kNearestNeighbors(points_[i], k_neighbors + 1);

            double sum_dist = 0.0;
            int count = 0;

            // Skip self (first nearest neighbor is self)
            for (size_t j = 1; j < neighbors.size(); ++j) {
                sum_dist += std::sqrt(neighbors[j].second);
                count++;
            }

            if (count > 0) {
                local_lengths[i] = sum_dist / count;
            }
        }

        return local_lengths;
    }
    
    // Get global characteristic length (combining multiple methods)
    // Combines results from different estimation methods for robustness
    double getGlobalCharacteristicLength() {
        // Method 1: Nearest neighbor estimation
        double method1 = estimateFromNearestNeighbors(500);
        
        // Method 2: Multi-scale estimation
        double method2 = estimateMultiScale(32);
        
        // Method 3: Density estimation using default search radius
        // First estimate a reasonable search radius
        double initial_radius = method1 * 2.0;
        double method3 = estimateFromLocalDensity(initial_radius);
        
        // Return robust average of three methods (remove outliers)
        thrust::host_vector<double> methods = {method1, method2, method3};
        thrust::sort(methods.begin(), methods.end());

        // Use median as final result
        return methods[1]; // median
    }
};

// ============================================
// Adaptive Field Interpolator (Using Statistical Characteristic Length)
// ============================================
// This class performs adaptive interpolation of magnetic field data using
// characteristic length scales for optimal search radius selection.
// It combines KD-tree queries with inverse distance weighting.

class AdaptiveFieldInterpolator {
private:
    thrust::host_vector<Point3D> points_;
    thrust::host_vector<std::array<double, 3>> field_values_;
    std::unique_ptr<KDTree> kdtree_;
    thrust::host_vector<double> local_char_lengths_;
    double global_char_length_;

    // Cache recent query results to improve performance for repeated interpolations
    mutable std::unordered_map<Point3D, std::array<double, 3>, Point3DHash> interpolation_cache_;

public:
    AdaptiveFieldInterpolator(const thrust::host_vector<Point3D>& points,
                            const thrust::host_vector<std::array<double, 3>>& field_values)
        : points_(points), field_values_(field_values) {
        
        if (points.size() != field_values.size()) {
            throw std::runtime_error("Points and field values must have same size");
        }
        
        // Build KD tree
        kdtree_ = std::make_unique<KDTree>(points_);
        
        // Calculate characteristic length
        StatisticalLengthEstimator estimator(points_);
        
        // Get global characteristic length
        global_char_length_ = estimator.getGlobalCharacteristicLength();
        std::cout << "Global characteristic length estimated: " << global_char_length_ << std::endl;
        
        // Calculate local characteristic length
        local_char_lengths_ = estimator.computeLocalCharacteristicLengths();
    }
    
    // Single point interpolation using adaptive search radius and inverse distance weighting
    std::array<double, 3> interpolate(const Point3D& query,
                                     double radius_factor = 2.0,
                                     int min_neighbors = 4) const {
        
        // Check cache for previously computed interpolation results
        auto cache_it = interpolation_cache_.find(query);
        if (cache_it != interpolation_cache_.end()) {
            return cache_it->second;
        }
        
        // Find nearest point to estimate local characteristic length
        auto nearest = kdtree_->nearestNeighbor(query);
        if (nearest.first == -1) {
            return {0.0, 0.0, 0.0};
        }
        
        double local_char_length;
        if (nearest.first < local_char_lengths_.size()) {
            local_char_length = local_char_lengths_[nearest.first];
        } else {
            local_char_length = global_char_length_;
        }
        
        // Adaptive search radius based on local characteristic length
        double search_radius = radius_factor * local_char_length;
        
        // Radius search
        auto indices = kdtree_->radiusSearch(query, search_radius);
        
        // If too few points found, expand search radius
        if (indices.size() < min_neighbors) {
            search_radius *= 1.5;
            indices = kdtree_->radiusSearch(query, search_radius);
        }
        
        // If still too few, use K nearest neighbors
        if (indices.size() < min_neighbors) {
            auto k_neighbors = kdtree_->kNearestNeighbors(query, min_neighbors * 2);
            indices.clear();
            for (const auto& [idx, dist] : k_neighbors) {
                indices.push_back(idx);
            }
        }
        
        // Inverse distance weighted interpolation
        std::array<double, 3> result = {0.0, 0.0, 0.0};
        double total_weight = 0.0;
        
        for (int idx : indices) {
            double dist = query.distance(points_[idx]);
            
            // Adaptive weight: consider local characteristic length
            double local_scale = local_char_lengths_[idx];
            double epsilon = local_scale * 0.1; // Prevent division by zero
            double weight = 1.0 / (dist * dist + epsilon * epsilon);
            
            for (int comp = 0; comp < 3; ++comp) {
                result[comp] += weight * field_values_[idx][comp];
            }
            total_weight += weight;
        }
        
        if (total_weight > 0) {
            for (int comp = 0; comp < 3; ++comp) {
                result[comp] /= total_weight;
            }
        }
        
        // Update cache (limit cache size)
        if (interpolation_cache_.size() < 1000) {
            interpolation_cache_[query] = result;
        }
        
        return result;
    }
    
    // Compute magnetic field gradient using central difference approximation
    std::array<std::array<double, 3>, 3> computeGradient(const Point3D& query) const {
        std::array<std::array<double, 3>, 3> gradient = {0.0};
        
        // Estimate local characteristic length
        auto nearest = kdtree_->nearestNeighbor(query);
        if (nearest.first == -1) return gradient;
        
        double h;
        if (nearest.first < local_char_lengths_.size()) {
            h = local_char_lengths_[nearest.first] * 0.1; // Differentiation step size
        } else {
            h = global_char_length_ * 0.1;
        }
        
        // Avoid step size too small
        h = std::max(h, 1e-6);
        
        // Central difference to calculate gradient
        for (int dim = 0; dim < 3; ++dim) {
            Point3D query_plus = query;
            Point3D query_minus = query;
            
            if (dim == 0) {
                query_plus.x += h;
                query_minus.x -= h;
            } else if (dim == 1) {
                query_plus.y += h;
                query_minus.y -= h;
            } else {
                query_plus.z += h;
                query_minus.z -= h;
            }
            
            auto B_plus = interpolate(query_plus);
            auto B_minus = interpolate(query_minus);
            
            for (int comp = 0; comp < 3; ++comp) {
                gradient[comp][dim] = (B_plus[comp] - B_minus[comp]) / (2.0 * h);
            }
        }
        
        return gradient;
    }
    
    double getGlobalCharacteristicLength() const {
        return global_char_length_;
    }
    
    const thrust::host_vector<double>& getLocalCharacteristicLengths() const {
        return local_char_lengths_;
    }
};

// ============================================
// Helper Functions: Generate Test Data
// ============================================
// Utility functions for creating synthetic point clouds and magnetic field data
// for testing and demonstration purposes.

thrust::host_vector<Point3D> generateTestPoints(int count, double domain_size = 10.0) {
    thrust::host_vector<Point3D> points;
    points.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, domain_size);
    
    for (int i = 0; i < count; ++i) {
        points.push_back(Point3D(dis(gen), dis(gen), dis(gen)));
    }
    
    return points;
}

thrust::host_vector<std::array<double, 3>> generateTestField(const thrust::host_vector<Point3D>& points) {
    thrust::host_vector<std::array<double, 3>> field_values;
    field_values.reserve(points.size());
    
    // Generate a simple magnetic field using a dipole model for testing
    Point3D dipole_center(5.0, 5.0, 5.0);
    std::array<double, 3> dipole_moment = {1.0, 0.0, 0.0}; // Dipole moment along x-direction
    
    for (const auto& point : points) {
        std::array<double, 3> B = {0.0, 0.0, 0.0};
        
        // Calculate vector to dipole center
        double dx = point.x - dipole_center.x;
        double dy = point.y - dipole_center.y;
        double dz = point.z - dipole_center.z;
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (r > 1e-6) {
            double r3 = r * r * r;
            double r5 = r3 * r * r;
            
            // Dipole magnetic field formula: B = (3(m·r)r - m) / r^5
            double m_dot_r = dipole_moment[0]*dx + dipole_moment[1]*dy + dipole_moment[2]*dz;
            
            B[0] = (3 * m_dot_r * dx - dipole_moment[0] * r*r) / r5;
            B[1] = (3 * m_dot_r * dy - dipole_moment[1] * r*r) / r5;
            B[2] = (3 * m_dot_r * dz - dipole_moment[2] * r*r) / r5;
        }
        
        field_values.push_back(B);
    }
    
    return field_values;
}

// ============================================
// Performance Testing and Demonstration
// ============================================
// This section contains the main demonstration function that showcases
// the capabilities of the 3D interpolation system through various tests.

void runDemo() {
    std::cout << "============================================" << std::endl;
    std::cout << "FEA-DEM Magnetic Field Interpolation Characteristic Length Estimation Demo" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Generate test data
    int point_count = 5000;
    std::cout << "Generating " << point_count << " test points..." << std::endl;
    
    auto points = generateTestPoints(point_count);
    auto field_values = generateTestField(points);
    
    std::cout << "Data generation completed" << std::endl;
    std::cout << std::endl;
    
    // Test characteristic length estimation
    {
        std::cout << "1. Characteristic Length Estimation Test" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        StatisticalLengthEstimator estimator(points);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        double method1 = estimator.estimateFromNearestNeighbors(500);
        double method2 = estimator.estimateMultiScale(32);
        double method3 = estimator.estimateFromLocalDensity(method1 * 2.0);
        double global_length = estimator.getGlobalCharacteristicLength();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        std::cout << "  Nearest neighbor method: " << method1 << std::endl;
        std::cout << "  Multi-scale method: " << method2 << std::endl;
        std::cout << "  Density method: " << method3 << std::endl;
        std::cout << "  Combined global characteristic length: " << global_length << std::endl;
        std::cout << "  Computation time: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    // Test interpolator
    {
        std::cout << "2. Adaptive Interpolator Test" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        AdaptiveFieldInterpolator interpolator(points, field_values);
        
        auto end_build = std::chrono::high_resolution_clock::now();
        
        // Test several query points
        thrust::host_vector<Point3D> query_points = {
            Point3D(5.0, 5.0, 5.0),
            Point3D(2.0, 3.0, 4.0),
            Point3D(7.0, 8.0, 6.0),
            Point3D(1.0, 1.0, 1.0),
            Point3D(9.0, 9.0, 9.0)
        };
        
        std::cout << "  Query point magnetic field interpolation results:" << std::endl;
        for (size_t i = 0; i < query_points.size(); ++i) {
            auto B = interpolator.interpolate(query_points[i]);
            std::cout << "    Point " << i+1 << " ("
                      << query_points[i].x << ", "
                      << query_points[i].y << ", "
                      << query_points[i].z << "): "
                      << "B = [" << B[0] << ", " << B[1] << ", " << B[2] << "]" << std::endl;
        }
        
        auto end_query = std::chrono::high_resolution_clock::now();
        
        std::cout << std::endl;
        std::cout << "  Interpolator build time: "
                  << std::chrono::duration<double, std::milli>(end_build - start).count()
                  << " ms" << std::endl;
        std::cout << "  5-point query time: "
                  << std::chrono::duration<double, std::milli>(end_query - end_build).count()
                  << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    // Test gradient computation
    {
        std::cout << "3. Magnetic Field Gradient Computation Test" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        AdaptiveFieldInterpolator interpolator(points, field_values);
        
        Point3D query_point(5.0, 5.0, 5.0);
        auto gradient = interpolator.computeGradient(query_point);
        
        std::cout << "  Magnetic field gradient at point (5, 5, 5):" << std::endl;
        std::cout << "    ∇Bx = [" << gradient[0][0] << ", " << gradient[0][1] << ", " << gradient[0][2] << "]" << std::endl;
        std::cout << "    ∇By = [" << gradient[1][0] << ", " << gradient[1][1] << ", " << gradient[1][2] << "]" << std::endl;
        std::cout << "    ∇Bz = [" << gradient[2][0] << ", " << gradient[2][1] << ", " << gradient[2][2] << "]" << std::endl;
        std::cout << std::endl;
    }
    
    // Performance benchmark
    {
        std::cout << "4. Performance Benchmark" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        AdaptiveFieldInterpolator interpolator(points, field_values);
        
        // Generate 100 random query points
        thrust::host_vector<Point3D> test_queries = generateTestPoints(100, 10.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& query : test_queries) {
            auto B = interpolator.interpolate(query);
            // Prevent compiler from optimizing away
            volatile double dummy = B[0] + B[1] + B[2];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        std::cout << "  100 interpolation query time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Average time per query: " << duration.count() / 100.0 << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "Demo completed" << std::endl;
    std::cout << "============================================" << std::endl;
}

// ============================================
// Main function
// ============================================

int main() {
    try {
        runDemo();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}