#include "tso/attention.hpp"
#include "tso/transformer.hpp"
#include "tso/model.hpp"
#include "tso/tokenizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tso;

void test_layernorm_finite_diff() {
    std::cout << "[Test 1] Testing LayerNorm Forward and Backward Gradient Check...\n";
    size_t dim = 8;
    LayerNorm ln(dim);
    ln.gamma = {1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.3, 0.7};
    ln.beta = {0.1, -0.2, 0.05, -0.1, 0.2, -0.05, 0.15, -0.15};

    Sequence X = {
        {0.5, -1.2, 2.0, 0.1, -0.4, 1.5, -0.8, 0.9},
        {-0.3, 0.8, -1.5, 2.2, 0.0, -0.7, 1.1, -1.0}
    };

    std::vector<std::vector<double>> xhat;
    std::vector<double> std_inv;
    Sequence Y = ln.forward(X, xhat, std_inv);

    assert(Y.size() == 2);
    assert(Y[0].size() == dim);

    // Verify forward normalization properties: mean ≈ 0, std ≈ 1 before affine
    for (size_t t = 0; t < 2; ++t) {
        double m = 0.0;
        for (double v : xhat[t]) m += v;
        m /= static_cast<double>(dim);
        assert(std::abs(m) < 1e-5);
        (void)m;
    }

    // Gradient check via finite differences on input X[0][2]
    Sequence dY = {
        {0.1, -0.3, 0.5, 0.2, -0.1, 0.4, -0.2, 0.3},
        {-0.2, 0.4, 0.1, -0.5, 0.3, 0.0, -0.1, 0.2}
    };

    ln.zero_grad();
    Sequence dX = ln.backward(dY, xhat, std_inv);

    // Compute numerical gradient for X[0][2]
    double eps = 1e-6;
    Sequence X_pos = X; X_pos[0][2] += eps;
    Sequence X_neg = X; X_neg[0][2] -= eps;

    std::vector<std::vector<double>> xhat_p, xhat_n;
    std::vector<double> std_inv_p, std_inv_n;
    Sequence Y_pos = ln.forward(X_pos, xhat_p, std_inv_p);
    Sequence Y_neg = ln.forward(X_neg, xhat_n, std_inv_n);

    double loss_pos = 0.0, loss_neg = 0.0;
    for (size_t t = 0; t < 2; ++t) {
        for (size_t d = 0; d < dim; ++d) {
            loss_pos += Y_pos[t][d] * dY[t][d];
            loss_neg += Y_neg[t][d] * dY[t][d];
        }
    }
    double num_grad = (loss_pos - loss_neg) / (2.0 * eps);
    double ana_grad = dX[0][2];

    assert(std::abs(num_grad - ana_grad) < 1e-4);
    (void)num_grad;
    (void)ana_grad;
    std::cout << "  LayerNorm Forward & Backward Gradient check PASSED (ana=" << ana_grad << ", num=" << num_grad << ")\n";
}

void test_multihead_attention() {
    std::cout << "[Test 2] Testing MultiHeadAttention Forward, Softmax Sum, and Backward...\n";
    size_t d_model = 16;
    size_t n_heads = 2;
    MultiHeadAttention mha(d_model, n_heads, 12345);

    size_t L = 6;
    Sequence X(L, std::vector<double>(d_model, 0.0));
    for (size_t i = 0; i < L; ++i) {
        for (size_t d = 0; d < d_model; ++d) {
            X[i][d] = std::sin(static_cast<double>(i * d_model + d));
        }
    }

    Sequence Y = mha.forward(X);
    assert(Y.size() == L);
    assert(Y[0].size() == d_model);

    // Verify Attention matrix row sum = 1.0 for all heads
    for (size_t h = 0; h < n_heads; ++h) {
        const auto& A = mha.get_attention_weights(h);
        assert(A.size() == L);
        for (size_t i = 0; i < L; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < L; ++j) {
                assert(A[i][j] >= 0.0 && A[i][j] <= 1.0);
                row_sum += A[i][j];
            }
            assert(std::abs(row_sum - 1.0) < 1e-5);
            (void)row_sum;
        }
    }

    // Backward pass check
    Sequence dY(L, std::vector<double>(d_model, 0.1));
    mha.zero_grad();
    Sequence dX = mha.backward(dY);
    assert(dX.size() == L);
    assert(dX[0].size() == d_model);

    // Finite difference check on input X[2][5]
    double eps = 1e-6;
    Sequence X_pos = X; X_pos[2][5] += eps;
    Sequence X_neg = X; X_neg[2][5] -= eps;

    Sequence Y_pos = mha.forward(X_pos);
    Sequence Y_neg = mha.forward(X_neg);

    double loss_pos = 0.0, loss_neg = 0.0;
    for (size_t i = 0; i < L; ++i) {
        for (size_t d = 0; d < d_model; ++d) {
            loss_pos += Y_pos[i][d] * dY[i][d];
            loss_neg += Y_neg[i][d] * dY[i][d];
        }
    }
    double num_grad = (loss_pos - loss_neg) / (2.0 * eps);
    double ana_grad = dX[2][5];

    assert(std::abs(num_grad - ana_grad) < 1e-3);
    std::cout << "  MultiHeadAttention Analytical vs Numerical dX PASSED (ana=" << ana_grad << ", num=" << num_grad << ")\n";
}

void test_transformer_block() {
    std::cout << "[Test 3] Testing TransformerEncoderBlock Full Forward & Backward Flow...\n";
    size_t d_model = 16;
    size_t n_heads = 2;
    size_t d_ff = 32;
    TransformerEncoderBlock block(d_model, n_heads, d_ff, 999);

    size_t L = 8;
    Sequence X(L, std::vector<double>(d_model, 0.0));
    for (size_t i = 0; i < L; ++i) {
        for (size_t d = 0; d < d_model; ++d) {
            X[i][d] = 0.1 * static_cast<double>(i + 1) * std::cos(static_cast<double>(d));
        }
    }

    Sequence Y = block.forward(X);
    assert(Y.size() == L);
    assert(Y[0].size() == d_model);

    Sequence dY(L, std::vector<double>(d_model, 0.05));
    block.zero_grad();
    Sequence dX = block.backward(dY);
    assert(dX.size() == L);
    assert(dX[0].size() == d_model);

    // Optimizer step
    block.step_adamw(0.001, 1);
    std::cout << "  TransformerEncoderBlock Forward, Backward & Optimizer Step PASSED\n";
}

void test_attention_multihead_mlp() {
    std::cout << "[Test 4] Testing AttentionMultiHeadMLP End-to-End Execution...\n";
    Random rng(42);
    StructuredState state{
        .declared = true,
        .registered = true,
        .runtime = RuntimeState::Running,
        .witness = WitnessState::Valid,
        .freshness = FreshnessState::Fresh,
        .health = HealthState::Healthy
    };

    auto tokens = Tokenizer::tokenize(state);
    assert(tokens.size() == 14);

    std::vector<LayerConfig> trunk_configs = {
        {16, 32, Activation::GELU},
        {32, 16, Activation::GELU}
    };

    AttentionMultiHeadMLP model(
        Tokenizer::vocab_size(),
        14,  // max_seq_len
        16,  // embedding_dim
        2,   // num_heads
        32,  // ff_dim
        trunk_configs,
        16,  // latent_dim
        4,   // choice_dim
        7,   // noul_dim
        rng
    );

    auto out = model.forward(tokens);
    assert(out.choice_probs.size() == 4);
    assert(out.noul_probs.size() == 7);

    // Backward pass
    Vector dchoice(4, 0.01f);
    Vector dnoul(7, 0.01f);
    Vector dscore(1, 0.05f);
    Vector dunc(1, 0.02f);

    model.zero_grad();
    model.backward(dchoice, dnoul, dscore, dunc);

    AdamW opt(0.001f);
    model.update(opt);

    std::cout << "  AttentionMultiHeadMLP Model Params: " << model.num_params() << " parameters\n";
    std::cout << "  AttentionMultiHeadMLP End-to-End PASSED\n";
}

int main() {
    std::cout << "=== Running TinySystemOne Attention & Transformer Tests ===\n";
    test_layernorm_finite_diff();
    test_multihead_attention();
    test_transformer_block();
    test_attention_multihead_mlp();
    std::cout << "=== ALL ATTENTION TESTS PASSED ===\n";
    return 0;
}
