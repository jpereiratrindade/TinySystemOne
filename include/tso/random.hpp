#pragma once

#include "tensor.hpp"
#include <random>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace tso {

class Random {
public:
    explicit Random(std::uint64_t seed = 42) : rng_(seed) {}

    void set_seed(std::uint64_t seed) {
        rng_.seed(seed);
    }

    Scalar uniform(Scalar min_val = 0.0f, Scalar max_val = 1.0f) {
        std::uniform_real_distribution<Scalar> dist(min_val, max_val);
        return dist(rng_);
    }

    int uniform_int(int min_val, int max_val) {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        return dist(rng_);
    }

    Scalar normal(Scalar mean = 0.0f, Scalar stddev = 1.0f) {
        std::normal_distribution<Scalar> dist(mean, stddev);
        return dist(rng_);
    }

    // He (Kaiming) Normal initialization: std = sqrt(2.0 / fan_in)
    void init_he_normal(Matrix& W) {
        const Scalar stddev = std::sqrt(2.0f / static_cast<Scalar>(W.cols));
        std::normal_distribution<Scalar> dist(0.0f, stddev);
        for (auto& val : W.data) {
            val = dist(rng_);
        }
    }

    // Xavier (Glorot) Uniform initialization: bound = sqrt(6 / (fan_in + fan_out))
    void init_xavier_uniform(Matrix& W) {
        const Scalar bound = std::sqrt(6.0f / static_cast<Scalar>(W.cols + W.rows));
        std::uniform_real_distribution<Scalar> dist(-bound, bound);
        for (auto& val : W.data) {
            val = dist(rng_);
        }
    }

    void init_zeros(Vector& v) {
        std::fill(v.begin(), v.end(), 0.0f);
    }

    template <typename T>
    void shuffle(std::vector<T>& vec) {
        std::shuffle(vec.begin(), vec.end(), rng_);
    }

private:
    std::mt19937_64 rng_;
};

} // namespace tso
