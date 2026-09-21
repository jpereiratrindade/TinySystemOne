#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <numbers>

namespace tso {

using Scalar = float;

enum class Activation {
    None,
    ReLU,
    GELU,
    Sigmoid,
    Softplus
};

struct Matrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<Scalar> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, Scalar initial = 0.0f)
        : rows(r), cols(c), data(r * c, initial) {}

    [[nodiscard]] inline Scalar& at(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    [[nodiscard]] inline const Scalar& at(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }

    [[nodiscard]] inline Scalar& operator()(std::size_t r, std::size_t c) {
        return data[r * cols + c];
    }

    [[nodiscard]] inline const Scalar& operator()(std::size_t r, std::size_t c) const {
        return data[r * cols + c];
    }

    void fill(Scalar val) {
        std::fill(data.begin(), data.end(), val);
    }
};

using Vector = std::vector<Scalar>;

// Matrix * Vector (y = W * x)
inline Vector matmul(const Matrix& W, const Vector& x) {
    if (W.cols != x.size()) {
        throw std::invalid_argument("Matrix cols must match Vector size in matmul");
    }
    Vector y(W.rows, 0.0f);
    for (std::size_t i = 0; i < W.rows; ++i) {
        Scalar sum = 0.0f;
        const std::size_t row_offset = i * W.cols;
        for (std::size_t j = 0; j < W.cols; ++j) {
            sum += W.data[row_offset + j] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

// Matrix^T * Vector (y = W^T * dy)
inline Vector matmul_transpose(const Matrix& W, const Vector& dy) {
    if (W.rows != dy.size()) {
        throw std::invalid_argument("Matrix rows must match dy size in matmul_transpose");
    }
    Vector dx(W.cols, 0.0f);
    for (std::size_t i = 0; i < W.rows; ++i) {
        const Scalar val = dy[i];
        const std::size_t row_offset = i * W.cols;
        for (std::size_t j = 0; j < W.cols; ++j) {
            dx[j] += W.data[row_offset + j] * val;
        }
    }
    return dx;
}

// Vector addition: a += b
inline void vec_add_(Vector& a, const Vector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vector sizes must match in vec_add_");
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
}

// Outer product accumulation: dW += dy * x^T
inline void outer_product_accumulate(Matrix& dW, const Vector& dy, const Vector& x) {
    if (dW.rows != dy.size() || dW.cols != x.size()) {
        throw std::invalid_argument("Dimension mismatch in outer_product_accumulate");
    }
    for (std::size_t i = 0; i < dW.rows; ++i) {
        const Scalar yi = dy[i];
        const std::size_t row_offset = i * dW.cols;
        for (std::size_t j = 0; j < dW.cols; ++j) {
            dW.data[row_offset + j] += yi * x[j];
        }
    }
}

// ReLU
inline Scalar relu(Scalar x) {
    return x > 0.0f ? x : 0.0f;
}

inline Scalar drelu(Scalar x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// GELU (approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))
inline Scalar gelu(Scalar x) {
    constexpr Scalar kSqrt2OverPi = 0.7978845608028654f;
    const Scalar cube = 0.044715f * x * x * x;
    const Scalar inner = kSqrt2OverPi * (x + cube);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

inline Scalar dgelu(Scalar x) {
    constexpr Scalar kSqrt2OverPi = 0.7978845608028654f;
    const Scalar cube = 0.044715f * x * x * x;
    const Scalar inner = kSqrt2OverPi * (x + cube);
    const Scalar tanh_val = std::tanh(inner);
    const Scalar sech2 = 1.0f - tanh_val * tanh_val;
    const Scalar d_inner = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
}

// Sigmoid
inline Scalar sigmoid(Scalar x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline Scalar dsigmoid(Scalar x) {
    const Scalar s = sigmoid(x);
    return s * (1.0f - s);
}

// Softplus: ln(1 + e^x), derivative is Sigmoid(x)
inline Scalar softplus(Scalar x) {
    if (x > 20.0f) return x; // Avoid overflow
    return std::log1p(std::exp(x));
}

inline Scalar dsoftplus(Scalar x) {
    return sigmoid(x);
}

// Apply activation to vector
inline Vector apply_activation(const Vector& x, Activation act) {
    Vector y(x.size());
    switch (act) {
        case Activation::ReLU:
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = relu(x[i]);
            break;
        case Activation::GELU:
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = gelu(x[i]);
            break;
        case Activation::Sigmoid:
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = sigmoid(x[i]);
            break;
        case Activation::Softplus:
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = softplus(x[i]);
            break;
        case Activation::None:
        default:
            y = x;
            break;
    }
    return y;
}

// Apply backward activation derivative elementwise: dx = dy * dAct(z)
inline Vector backward_activation(const Vector& dy, const Vector& z, Activation act) {
    Vector dx(dy.size());
    switch (act) {
        case Activation::ReLU:
            for (std::size_t i = 0; i < dy.size(); ++i) dx[i] = dy[i] * drelu(z[i]);
            break;
        case Activation::GELU:
            for (std::size_t i = 0; i < dy.size(); ++i) dx[i] = dy[i] * dgelu(z[i]);
            break;
        case Activation::Sigmoid:
            for (std::size_t i = 0; i < dy.size(); ++i) dx[i] = dy[i] * dsigmoid(z[i]);
            break;
        case Activation::Softplus:
            for (std::size_t i = 0; i < dy.size(); ++i) dx[i] = dy[i] * dsoftplus(z[i]);
            break;
        case Activation::None:
        default:
            dx = dy;
            break;
    }
    return dx;
}

// Numerically stable Softmax
inline Vector softmax(const Vector& logits) {
    if (logits.empty()) return {};
    const Scalar max_val = *std::max_element(logits.begin(), logits.end());
    Vector probs(logits.size());
    Scalar sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    const Scalar inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs[i] *= inv_sum;
    }
    return probs;
}

} // namespace tso
