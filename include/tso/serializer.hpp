#pragma once

#include "model.hpp"
#include "calibration.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <chrono>

namespace tso {

struct ModelIdentity {
    std::string model_name{"TinySystemOne-MultiHead"};
    std::string version{"0.5.0"};
    std::uint64_t trained_timestamp{0};
    std::size_t parameter_count{0};
    Scalar calibrated_temperature{1.0f};
    Scalar ood_energy_threshold{0.0f};
};

class Serializer {
public:
    static constexpr std::uint32_t kMagicNumber = 0x54534F31; // "TSO1"
    static constexpr std::uint32_t kFormatVersion = 1;

    static void save(
        const std::filesystem::path& path,
        const MultiHeadMLP& model,
        const TemperatureScaler& scaler,
        const SelectivePredictor& predictor,
        const ModelIdentity& identity = {}
    ) {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error(std::format("Failed to open file for writing: {}", path.string()));
        }

        // 1. Magic & Version Header
        out.write(reinterpret_cast<const char*>(&kMagicNumber), sizeof(kMagicNumber));
        out.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));

        // 2. Model Identity Metadata
        write_string(out, identity.model_name);
        write_string(out, identity.version);
        const std::uint64_t timestamp = identity.trained_timestamp != 0 
            ? identity.trained_timestamp 
            : static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
        out.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
        
        const Scalar temp = scaler.temperature;
        const Scalar energy_tau = predictor.energy_threshold;
        out.write(reinterpret_cast<const char*>(&temp), sizeof(temp));
        out.write(reinterpret_cast<const char*>(&energy_tau), sizeof(energy_tau));

        // 3. Model Architecture Topology
        const auto& trunk = const_cast<MultiHeadMLP&>(model).trunk();
        const std::uint32_t num_trunk_layers = static_cast<std::uint32_t>(trunk.size());
        out.write(reinterpret_cast<const char*>(&num_trunk_layers), sizeof(num_trunk_layers));

        for (const auto& layer : trunk) {
            write_layer(out, layer);
        }

        write_layer(out, const_cast<MultiHeadMLP&>(model).choice_head());
        write_layer(out, const_cast<MultiHeadMLP&>(model).noul_head());
        write_layer(out, const_cast<MultiHeadMLP&>(model).score_head());
        write_layer(out, const_cast<MultiHeadMLP&>(model).uncertainty_head());
    }

    static void load(
        const std::filesystem::path& path,
        MultiHeadMLP& model,
        TemperatureScaler& scaler,
        SelectivePredictor& predictor,
        ModelIdentity& identity
    ) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error(std::format("Failed to open model file for reading: {}", path.string()));
        }

        // 1. Validate Header
        std::uint32_t magic = 0;
        std::uint32_t version = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != kMagicNumber) {
            throw std::runtime_error(std::format("Invalid model file magic header: 0x{:08X}", magic));
        }
        if (version != kFormatVersion) {
            throw std::runtime_error(std::format("Unsupported model format version: {}", version));
        }

        // 2. Metadata
        identity.model_name = read_string(in);
        identity.version = read_string(in);
        in.read(reinterpret_cast<char*>(&identity.trained_timestamp), sizeof(identity.trained_timestamp));
        
        Scalar temp = 1.0f;
        Scalar energy_tau = 0.0f;
        in.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        in.read(reinterpret_cast<char*>(&energy_tau), sizeof(energy_tau));
        
        scaler.temperature = temp;
        predictor.energy_threshold = energy_tau;
        identity.calibrated_temperature = temp;
        identity.ood_energy_threshold = energy_tau;

        // 3. Topology & Weights
        std::uint32_t num_trunk_layers = 0;
        in.read(reinterpret_cast<char*>(&num_trunk_layers), sizeof(num_trunk_layers));

        auto& trunk = model.trunk();
        if (trunk.size() != num_trunk_layers) {
            throw std::runtime_error(std::format("Mismatch in trunk layers: expected {}, found {}", trunk.size(), num_trunk_layers));
        }

        for (auto& layer : trunk) {
            read_layer(in, layer);
        }

        read_layer(in, model.choice_head());
        read_layer(in, model.noul_head());
        read_layer(in, model.score_head());
        read_layer(in, model.uncertainty_head());

        identity.parameter_count = model.num_params();
    }

private:
    static void write_string(std::ofstream& out, const std::string& str) {
        const std::uint32_t len = static_cast<std::uint32_t>(str.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len > 0) {
            out.write(str.data(), static_cast<std::streamsize>(len));
        }
    }

    static std::string read_string(std::ifstream& in) {
        std::uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (len == 0) return "";
        std::string str(len, '\0');
        in.read(str.data(), static_cast<std::streamsize>(len));
        return str;
    }

    static void write_layer(std::ofstream& out, const Layer& layer) {
        const std::uint32_t in_features = static_cast<std::uint32_t>(layer.W.cols);
        const std::uint32_t out_features = static_cast<std::uint32_t>(layer.W.rows);
        out.write(reinterpret_cast<const char*>(&in_features), sizeof(in_features));
        out.write(reinterpret_cast<const char*>(&out_features), sizeof(out_features));

        const auto& w_data = layer.W.data;
        const auto& b_data = layer.b;
        
        const std::uint32_t w_count = static_cast<std::uint32_t>(w_data.size());
        const std::uint32_t b_count = static_cast<std::uint32_t>(b_data.size());
        out.write(reinterpret_cast<const char*>(&w_count), sizeof(w_count));
        out.write(reinterpret_cast<const char*>(w_data.data()), static_cast<std::streamsize>(w_count * sizeof(Scalar)));

        out.write(reinterpret_cast<const char*>(&b_count), sizeof(b_count));
        out.write(reinterpret_cast<const char*>(b_data.data()), static_cast<std::streamsize>(b_count * sizeof(Scalar)));
    }

    static void read_layer(std::ifstream& in, Layer& layer) {
        std::uint32_t in_features = 0;
        std::uint32_t out_features = 0;
        in.read(reinterpret_cast<char*>(&in_features), sizeof(in_features));
        in.read(reinterpret_cast<char*>(&out_features), sizeof(out_features));

        if (layer.W.cols != in_features || layer.W.rows != out_features) {
            throw std::runtime_error(std::format("Layer shape mismatch: expected ({}, {}), got ({}, {})",
                                                 layer.W.rows, layer.W.cols, out_features, in_features));
        }

        std::uint32_t w_count = 0;
        std::uint32_t b_count = 0;
        in.read(reinterpret_cast<char*>(&w_count), sizeof(w_count));
        in.read(reinterpret_cast<char*>(layer.W.data.data()), static_cast<std::streamsize>(w_count * sizeof(Scalar)));

        in.read(reinterpret_cast<char*>(&b_count), sizeof(b_count));
        in.read(reinterpret_cast<char*>(layer.b.data()), static_cast<std::streamsize>(b_count * sizeof(Scalar)));
    }
};

} // namespace tso
