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

// ============================================
// 基础数据结构
// ============================================

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
    
    // 用于哈希映射
    bool operator==(const Point3D& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// 哈希函数用于Point3D
struct Point3DHash {
    std::size_t operator()(const Point3D& p) const {
        std::size_t h1 = std::hash<double>{}(p.x);
        std::size_t h2 = std::hash<double>{}(p.y);
        std::size_t h3 = std::hash<double>{}(p.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// 磁场数据
struct FieldData {
    Point3D position;
    std::array<double, 3> B_field;  // Bx, By, Bz
};

// ============================================
// KD树实现（用于快速最近邻搜索）
// ============================================

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
    std::vector<Point3D> points_;
    
    Node* buildTree(std::vector<std::pair<Point3D, int>>& points_vec, 
                   int start, int end, int depth) {
        if (start >= end) return nullptr;
        
        int axis = depth % 3;
        
        // 按当前轴排序
        std::sort(points_vec.begin() + start, points_vec.begin() + end,
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
        
        // 检查是否需要搜索另一边
        if (heap.size() < k || diff * diff < heap.top().first) {
            kNearestNeighbors(second, query, heap, k);
        }
    }
    
    void radiusSearch(Node* node, const Point3D& query, double radius_sq,
                     std::vector<int>& results) const {
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
    KDTree(const std::vector<Point3D>& points) : points_(points) {
        std::vector<std::pair<Point3D, int>> points_vec;
        points_vec.reserve(points.size());
        
        for (size_t i = 0; i < points.size(); ++i) {
            points_vec.emplace_back(points[i], i);
        }
        
        root_ = buildTree(points_vec, 0, points.size(), 0);
    }
    
    ~KDTree() {
        // 递归删除节点
        std::function<void(Node*)> deleteNode = [&](Node* node) {
            if (!node) return;
            deleteNode(node->left);
            deleteNode(node->right);
            delete node;
        };
        deleteNode(root_);
    }
    
    // K最近邻搜索
    std::vector<std::pair<int, double>> kNearestNeighbors(const Point3D& query, int k) const {
        std::priority_queue<std::pair<double, int>> heap; // 最大堆
        
        kNearestNeighbors(root_, query, heap, k);
        
        std::vector<std::pair<int, double>> results;
        while (!heap.empty()) {
            results.emplace_back(heap.top().second, heap.top().first);
            heap.pop();
        }
        std::reverse(results.begin(), results.end()); // 转为距离由小到大
        
        return results;
    }
    
    // 半径搜索
    std::vector<int> radiusSearch(const Point3D& query, double radius) const {
        std::vector<int> results;
        double radius_sq = radius * radius;
        radiusSearch(root_, query, radius_sq, results);
        return results;
    }
    
    // 最近邻搜索（单点）
    std::pair<int, double> nearestNeighbor(const Point3D& query) const {
        auto neighbors = kNearestNeighbors(query, 1);
        if (neighbors.empty()) return {-1, 0.0};
        return neighbors[0];
    }
};

// ============================================
// 基于点云统计的特征尺寸估计器
// ============================================

class StatisticalLengthEstimator {
private:
    std::vector<Point3D> points_;
    std::unique_ptr<KDTree> kdtree_;
    
public:
    StatisticalLengthEstimator(const std::vector<Point3D>& points) 
        : points_(points) {
        kdtree_ = std::make_unique<KDTree>(points_);
    }
    
    // 方法1：基于最近邻距离的简单估计
    double estimateFromNearestNeighbors(int sample_count = 1000) {
        if (points_.empty()) return 0.0;
        
        double sum_distances = 0.0;
        int actual_samples = std::min(sample_count, static_cast<int>(points_.size()));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, points_.size() - 1);
        
        for (int i = 0; i < actual_samples; ++i) {
            int idx = dis(gen);
            auto neighbors = kdtree_->kNearestNeighbors(points_[idx], 2); // 自身+最近邻
            if (neighbors.size() > 1) {
                sum_distances += std::sqrt(neighbors[1].second);
            }
        }
        
        return sum_distances / actual_samples;
    }
    
    // 方法2：基于局部密度估计
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
            
            // 排除自身
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
        
        // 特征长度 ∝ 密度^(-1/3)
        return std::pow(1.0 / avg_density, 1.0/3.0);
    }
    
    // 方法3：多尺度统计估计（最稳健）
    double estimateMultiScale(int max_k = 50) {
        if (points_.empty()) return 0.0;
        
        std::vector<double> characteristic_lengths;
        
        // 在不同尺度下计算特征长度
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
                    // 第k个最近邻的距离
                    sum_kdist += std::sqrt(neighbors[k].second);
                    valid_samples++;
                }
            }
            
            if (valid_samples > 0) {
                characteristic_lengths.push_back(sum_kdist / valid_samples);
            }
        }
        
        if (characteristic_lengths.empty()) return 0.0;
        
        // 取中位数作为最终估计（对异常值更稳健）
        std::sort(characteristic_lengths.begin(), characteristic_lengths.end());
        return characteristic_lengths[characteristic_lengths.size() / 2];
    }
    
    // 方法4：计算每个点的局部特征长度
    std::vector<double> computeLocalCharacteristicLengths(int k_neighbors = 6) {
        std::vector<double> local_lengths(points_.size(), 0.0);
        
        #pragma omp parallel for
        for (size_t i = 0; i < points_.size(); ++i) {
            auto neighbors = kdtree_->kNearestNeighbors(points_[i], k_neighbors + 1);
            
            double sum_dist = 0.0;
            int count = 0;
            
            // 跳过自身（第一个最近邻是自身）
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
    
    // 获取全局特征尺寸（综合多种方法）
    double getGlobalCharacteristicLength() {
        // 方法1：最近邻估计
        double method1 = estimateFromNearestNeighbors(500);
        
        // 方法2：多尺度估计
        double method2 = estimateMultiScale(32);
        
        // 方法3：使用默认搜索半径的密度估计
        // 先估算一个合理的搜索半径
        double initial_radius = method1 * 2.0;
        double method3 = estimateFromLocalDensity(initial_radius);
        
        // 返回三种方法的稳健平均值（去除异常值）
        std::vector<double> methods = {method1, method2, method3};
        std::sort(methods.begin(), methods.end());
        
        // 使用中位数作为最终结果
        return methods[1]; // 中位数
    }
};

// ============================================
// 自适应插值器（使用统计特征尺寸）
// ============================================

class AdaptiveFieldInterpolator {
private:
    std::vector<Point3D> points_;
    std::vector<std::array<double, 3>> field_values_;
    std::unique_ptr<KDTree> kdtree_;
    std::vector<double> local_char_lengths_;
    double global_char_length_;
    
    // 缓存最近查询结果
    mutable std::unordered_map<Point3D, std::array<double, 3>, Point3DHash> interpolation_cache_;
    
public:
    AdaptiveFieldInterpolator(const std::vector<Point3D>& points,
                            const std::vector<std::array<double, 3>>& field_values)
        : points_(points), field_values_(field_values) {
        
        if (points.size() != field_values.size()) {
            throw std::runtime_error("Points and field values must have same size");
        }
        
        // 构建KD树
        kdtree_ = std::make_unique<KDTree>(points_);
        
        // 计算特征尺寸
        StatisticalLengthEstimator estimator(points_);
        
        // 获取全局特征尺寸
        global_char_length_ = estimator.getGlobalCharacteristicLength();
        std::cout << "Global characteristic length estimated: " << global_char_length_ << std::endl;
        
        // 计算局部特征尺寸
        local_char_lengths_ = estimator.computeLocalCharacteristicLengths();
    }
    
    // 单点插值
    std::array<double, 3> interpolate(const Point3D& query, 
                                     double radius_factor = 2.0,
                                     int min_neighbors = 4) const {
        
        // 检查缓存
        auto cache_it = interpolation_cache_.find(query);
        if (cache_it != interpolation_cache_.end()) {
            return cache_it->second;
        }
        
        // 找到最近点估计局部特征尺寸
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
        
        // 自适应搜索半径
        double search_radius = radius_factor * local_char_length;
        
        // 半径搜索
        auto indices = kdtree_->radiusSearch(query, search_radius);
        
        // 如果找到的点太少，扩大搜索半径
        if (indices.size() < min_neighbors) {
            search_radius *= 1.5;
            indices = kdtree_->radiusSearch(query, search_radius);
        }
        
        // 如果还是太少，使用K近邻
        if (indices.size() < min_neighbors) {
            auto k_neighbors = kdtree_->kNearestNeighbors(query, min_neighbors * 2);
            indices.clear();
            for (const auto& [idx, dist] : k_neighbors) {
                indices.push_back(idx);
            }
        }
        
        // 逆距离加权插值
        std::array<double, 3> result = {0.0, 0.0, 0.0};
        double total_weight = 0.0;
        
        for (int idx : indices) {
            double dist = query.distance(points_[idx]);
            
            // 自适应权重：考虑局部特征尺寸
            double local_scale = local_char_lengths_[idx];
            double epsilon = local_scale * 0.1; // 防止除零
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
        
        // 更新缓存（限制缓存大小）
        if (interpolation_cache_.size() < 1000) {
            interpolation_cache_[query] = result;
        }
        
        return result;
    }
    
    // 计算磁场梯度（使用中心差分）
    std::array<std::array<double, 3>, 3> computeGradient(const Point3D& query) const {
        std::array<std::array<double, 3>, 3> gradient = {0.0};
        
        // 估计局部特征尺寸
        auto nearest = kdtree_->nearestNeighbor(query);
        if (nearest.first == -1) return gradient;
        
        double h;
        if (nearest.first < local_char_lengths_.size()) {
            h = local_char_lengths_[nearest.first] * 0.1; // 差分步长
        } else {
            h = global_char_length_ * 0.1;
        }
        
        // 避免步长过小
        h = std::max(h, 1e-6);
        
        // 中心差分计算梯度
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
    
    const std::vector<double>& getLocalCharacteristicLengths() const {
        return local_char_lengths_;
    }
};

// ============================================
// 辅助函数：生成测试数据
// ============================================

std::vector<Point3D> generateTestPoints(int count, double domain_size = 10.0) {
    std::vector<Point3D> points;
    points.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, domain_size);
    
    for (int i = 0; i < count; ++i) {
        points.emplace_back(dis(gen), dis(gen), dis(gen));
    }
    
    return points;
}

std::vector<std::array<double, 3>> generateTestField(const std::vector<Point3D>& points) {
    std::vector<std::array<double, 3>> field_values;
    field_values.reserve(points.size());
    
    // 生成一个简单的磁场：偶极子场
    Point3D dipole_center(5.0, 5.0, 5.0);
    std::array<double, 3> dipole_moment = {1.0, 0.0, 0.0}; // 沿x方向的偶极矩
    
    for (const auto& point : points) {
        std::array<double, 3> B = {0.0, 0.0, 0.0};
        
        // 计算到偶极子中心的向量
        double dx = point.x - dipole_center.x;
        double dy = point.y - dipole_center.y;
        double dz = point.z - dipole_center.z;
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (r > 1e-6) {
            double r3 = r * r * r;
            double r5 = r3 * r * r;
            
            // 偶极子磁场公式: B = (3(m·r)r - m) / r^5
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
// 性能测试和演示
// ============================================

void runDemo() {
    std::cout << "============================================" << std::endl;
    std::cout << "FEA-DEM 磁场插值特征尺寸估计演示" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // 生成测试数据
    int point_count = 5000;
    std::cout << "生成 " << point_count << " 个测试点..." << std::endl;
    
    auto points = generateTestPoints(point_count);
    auto field_values = generateTestField(points);
    
    std::cout << "数据生成完成" << std::endl;
    std::cout << std::endl;
    
    // 测试特征尺寸估计
    {
        std::cout << "1. 特征尺寸估计测试" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        StatisticalLengthEstimator estimator(points);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        double method1 = estimator.estimateFromNearestNeighbors(500);
        double method2 = estimator.estimateMultiScale(32);
        double method3 = estimator.estimateFromLocalDensity(method1 * 2.0);
        double global_length = estimator.getGlobalCharacteristicLength();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        std::cout << "  最近邻方法: " << method1 << std::endl;
        std::cout << "  多尺度方法: " << method2 << std::endl;
        std::cout << "  密度方法: " << method3 << std::endl;
        std::cout << "  综合全局特征尺寸: " << global_length << std::endl;
        std::cout << "  计算时间: " << duration.count() << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    // 测试插值器
    {
        std::cout << "2. 自适应插值器测试" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        AdaptiveFieldInterpolator interpolator(points, field_values);
        
        auto end_build = std::chrono::high_resolution_clock::now();
        
        // 测试几个查询点
        std::vector<Point3D> query_points = {
            Point3D(5.0, 5.0, 5.0),
            Point3D(2.0, 3.0, 4.0),
            Point3D(7.0, 8.0, 6.0),
            Point3D(1.0, 1.0, 1.0),
            Point3D(9.0, 9.0, 9.0)
        };
        
        std::cout << "  查询点磁场插值结果:" << std::endl;
        for (size_t i = 0; i < query_points.size(); ++i) {
            auto B = interpolator.interpolate(query_points[i]);
            std::cout << "    点" << i+1 << " (" 
                     << query_points[i].x << ", " 
                     << query_points[i].y << ", "
                     << query_points[i].z << "): "
                     << "B = [" << B[0] << ", " << B[1] << ", " << B[2] << "]" << std::endl;
        }
        
        auto end_query = std::chrono::high_resolution_clock::now();
        
        std::cout << std::endl;
        std::cout << "  插值器构建时间: " 
                  << std::chrono::duration<double, std::milli>(end_build - start).count() 
                  << " ms" << std::endl;
        std::cout << "  5个点查询时间: " 
                  << std::chrono::duration<double, std::milli>(end_query - end_build).count() 
                  << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    // 测试梯度计算
    {
        std::cout << "3. 磁场梯度计算测试" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        AdaptiveFieldInterpolator interpolator(points, field_values);
        
        Point3D query_point(5.0, 5.0, 5.0);
        auto gradient = interpolator.computeGradient(query_point);
        
        std::cout << "  在点 (5, 5, 5) 处的磁场梯度:" << std::endl;
        std::cout << "    ∇Bx = [" << gradient[0][0] << ", " << gradient[0][1] << ", " << gradient[0][2] << "]" << std::endl;
        std::cout << "    ∇By = [" << gradient[1][0] << ", " << gradient[1][1] << ", " << gradient[1][2] << "]" << std::endl;
        std::cout << "    ∇Bz = [" << gradient[2][0] << ", " << gradient[2][1] << ", " << gradient[2][2] << "]" << std::endl;
        std::cout << std::endl;
    }
    
    // 性能基准测试
    {
        std::cout << "4. 性能基准测试" << std::endl;
        std::cout << "---------------------" << std::endl;
        
        AdaptiveFieldInterpolator interpolator(points, field_values);
        
        // 生成100个随机查询点
        std::vector<Point3D> test_queries = generateTestPoints(100, 10.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& query : test_queries) {
            auto B = interpolator.interpolate(query);
            // 防止编译器优化掉
            volatile double dummy = B[0] + B[1] + B[2];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        
        std::cout << "  100次插值查询时间: " << duration.count() << " ms" << std::endl;
        std::cout << "  平均每次查询时间: " << duration.count() / 100.0 << " ms" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "演示完成" << std::endl;
    std::cout << "============================================" << std::endl;
}

// ============================================
// 主函数
// ============================================

int main() {
    try {
        runDemo();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}