#include <tso/tensor.hpp>
#include <tso/random.hpp>
#include <tso/loss.hpp>
#include <tso/model.hpp>
#include <tso/optimizer.hpp>
#include <tso/calibration.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>

#define TSO_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "Assertion failed: " #expr " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

void test_matrix_vector_ops() {
    std::cout << "[TEST] Running test_matrix_vector_ops...\n";
    tso::Matrix W(2, 3);
    W(0, 0) = 1; W(0, 1) = 2; W(0, 2) = 3;
    W(1, 0) = 4; W(1, 1) = 5; W(1, 2) = 6;

    tso::Vector x = {1.0f, 2.0f, 3.0f};
    tso::Vector y = tso::matmul(W, x);

    TSO_ASSERT(std::abs(y[0] - 14.0f) < 1e-5f);
    TSO_ASSERT(std::abs(y[1] - 32.0f) < 1e-5f);

    tso::Vector dy = {2.0f, 1.0f};
    tso::Vector dx = tso::matmul_transpose(W, dy);
    TSO_ASSERT(std::abs(dx[0] - 6.0f) < 1e-5f);
    TSO_ASSERT(std::abs(dx[1] - 9.0f) < 1e-5f);
    TSO_ASSERT(std::abs(dx[2] - 12.0f) < 1e-5f);
    std::cout << "  ✓ Matrix vector operations passed!\n";
}

void test_softmax_and_loss() {
    std::cout << "[TEST] Running test_softmax_and_loss...\n";
    tso::Vector logits = {2.0f, 1.0f, 0.1f};
    tso::Vector probs = tso::softmax(logits);

    float sum = 0.0f;
    for (float p : probs) {
        sum += p;
        TSO_ASSERT(p > 0.0f && p < 1.0f);
    }
    TSO_ASSERT(std::abs(sum - 1.0f) < 1e-5f);
    TSO_ASSERT(probs[0] > probs[1] && probs[1] > probs[2]);

    auto [loss, dlogits] = tso::CrossEntropyLoss::compute_from_index(probs, 0);
    TSO_ASSERT(loss > 0.0f);
    TSO_ASSERT(dlogits.size() == 3);
    TSO_ASSERT(std::abs(dlogits[0] - (probs[0] - 1.0f)) < 1e-5f);
    TSO_ASSERT(std::abs(dlogits[1] - (probs[1] - 0.0f)) < 1e-5f);
    std::cout << "  ✓ Softmax and Loss passed!\n";
}

void test_gradient_check() {
    std::cout << "[TEST] Running numerical vs analytical gradient check (Single MLP)...\n";
    tso::Random rng(1234);
    std::vector<tso::LayerConfig> arch = {
        {4, 8, tso::Activation::GELU},
        {8, 3, tso::Activation::None}
    };
    tso::MLP mlp(arch, rng);

    tso::Vector x = {0.5f, -0.2f, 1.0f, 0.1f};
    std::size_t target_idx = 1;

    mlp.zero_grad();
    tso::Vector probs = mlp.forward(x);
    auto [loss, dlogits] = tso::CrossEntropyLoss::compute_from_index(probs, target_idx);
    mlp.backward(dlogits);

    constexpr float eps = 1e-4f;
    auto& layer0 = mlp.layers()[0];

    for (std::size_t r = 0; r < layer0.W.rows; ++r) {
        for (std::size_t c = 0; c < layer0.W.cols; ++c) {
            const float orig_w = layer0.W(r, c);

            layer0.W(r, c) = orig_w + eps;
            tso::Vector probs_plus = mlp.forward(x);
            float loss_plus = tso::CrossEntropyLoss::compute_from_index(probs_plus, target_idx).loss;

            layer0.W(r, c) = orig_w - eps;
            tso::Vector probs_minus = mlp.forward(x);
            float loss_minus = tso::CrossEntropyLoss::compute_from_index(probs_minus, target_idx).loss;

            layer0.W(r, c) = orig_w; // restore

            const float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
            const float anal_grad = layer0.dW(r, c);

            const float diff = std::abs(num_grad - anal_grad);
            TSO_ASSERT(diff < 1e-3f);
        }
    }
    std::cout << "  ✓ Single MLP gradient check passed!\n";
}

void test_multi_head_gradient_check() {
    std::cout << "[TEST] Running numerical vs analytical gradient check (4-Head MultiHeadMLP)...\n";
    tso::Random rng(5678);
    std::vector<tso::LayerConfig> trunk = {
        {4, 6, tso::Activation::GELU}
    };
    tso::MultiHeadMLP model(trunk, 6, 3, 4, rng);

    tso::Vector x = {0.3f, -0.7f, 0.2f, 0.9f};
    std::size_t c_target = 1;
    std::size_t n_target = 2;
    float s_target = 0.75f;

    auto eval_loss = [&](tso::MultiHeadMLP& m) -> float {
        auto out = m.forward(x);
        float cl = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, c_target).loss;
        float nl = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, n_target).loss;
        auto nll = tso::GaussianNLLLoss::compute(out.score, out.uncertainty, s_target);
        return cl + nl + nll.loss;
    };

    model.zero_grad();
    auto out = model.forward(x);
    auto [c_loss, d_c] = tso::CrossEntropyLoss::compute_from_index(out.choice_probs, c_target);
    auto [n_loss, d_n] = tso::CrossEntropyLoss::compute_from_index(out.noul_probs, n_target);
    auto nll = tso::GaussianNLLLoss::compute(out.score, out.uncertainty, s_target);
    model.backward(d_c, d_n, nll.d_mu, nll.d_var);

    constexpr float eps = 1e-3f;
    auto& trunk_layer = model.trunk()[0];

    for (std::size_t r = 0; r < trunk_layer.W.rows; ++r) {
        for (std::size_t c = 0; c < trunk_layer.W.cols; ++c) {
            const float orig_w = trunk_layer.W(r, c);

            trunk_layer.W(r, c) = orig_w + eps;
            float l_plus = eval_loss(model);

            trunk_layer.W(r, c) = orig_w - eps;
            float l_minus = eval_loss(model);

            trunk_layer.W(r, c) = orig_w;

            const float num_grad = (l_plus - l_minus) / (2.0f * eps);
            const float anal_grad = trunk_layer.dW(r, c);

            const float diff = std::abs(num_grad - anal_grad);
            TSO_ASSERT(diff < 2e-3f);
        }
    }
    std::cout << "  ✓ 4-Head MultiHeadMLP joint gradient check passed!\n";
}

void test_calibration_metrics() {
    std::cout << "[TEST] Running test_calibration_metrics...\n";
    tso::Vector uniform_p = {0.25f, 0.25f, 0.25f, 0.25f};
    tso::Vector certain_p = {0.999f, 0.0003f, 0.0003f, 0.0004f};

    float u_ent = tso::Calibration::normalized_entropy(uniform_p);
    float c_ent = tso::Calibration::normalized_entropy(certain_p);

    TSO_ASSERT(std::abs(u_ent - 1.0f) < 1e-4f);
    TSO_ASSERT(c_ent < 0.05f);

    std::vector<tso::Vector> all_probs = {
        {0.9f, 0.1f},
        {0.8f, 0.2f},
        {0.4f, 0.6f}
    };
    std::vector<std::size_t> all_targets = {0, 0, 1};

    auto report = tso::Calibration::evaluate(all_probs, all_targets, 5);
    TSO_ASSERT(std::abs(report.accuracy - 1.0f) < 1e-5f);
    TSO_ASSERT(report.ece >= 0.0f);
    TSO_ASSERT(report.brier_score > 0.0f);
    std::cout << "  ✓ Calibration metrics passed!\n";
}

int main() {
    std::cout << "=== TinySystemOne Math & Engine Unit Tests ===\n";
    test_matrix_vector_ops();
    test_softmax_and_loss();
    test_gradient_check();
    test_multi_head_gradient_check();
    test_calibration_metrics();
    std::cout << "All math tests passed successfully!\n";
    return 0;
}
