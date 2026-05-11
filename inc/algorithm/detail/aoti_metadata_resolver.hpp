#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <c10/core/DeviceType.h>

namespace algorithm::detail {

struct TypedPackagePaths {
    std::string cpu_f32_path;
    std::string cpu_f64_path;
    std::string cuda_f32_path;
    std::string cuda_f64_path;

    template <typename T>
    const std::string &SelectPath(c10::DeviceType device) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "TypedPackagePaths supports float and double only");

        const std::string *selected = nullptr;
        switch (device) {
        case c10::DeviceType::CPU:
            if constexpr (std::is_same_v<T, float>) {
                selected = &cpu_f32_path;
            } else {
                selected = &cpu_f64_path;
            }
            break;
        case c10::DeviceType::CUDA:
            if constexpr (std::is_same_v<T, float>) {
                selected = &cuda_f32_path;
            } else {
                selected = &cuda_f64_path;
            }
            break;
        default:
            throw std::runtime_error("Unsupported device type for AOT package "
                                     "selection");
        }

        if (selected == nullptr || selected->empty()) {
            throw std::runtime_error("Missing AOT package path for requested "
                                     "dtype/device variant");
        }
        if (!std::filesystem::exists(*selected)) {
            throw std::runtime_error("AOT package file does not exist: " +
                                     *selected);
        }
        return *selected;
    }
};

inline std::unordered_map<std::string, std::string>
LoadMetadataMap(const std::string &metadata_path) {
    std::ifstream in(metadata_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open AOTI metadata file: " +
                                 metadata_path);
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string line;
    int64_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }

        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Invalid metadata line in " +
                                     metadata_path + " at line " +
                                     std::to_string(line_no));
        }

        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "Invalid metadata entry in file: " + metadata_path +
                " at line " + std::to_string(line_no));
        }
        metadata[key] = value;
    }

    if (metadata.empty()) {
        throw std::runtime_error(
            "No metadata entries found in metadata file: " + metadata_path);
    }

    return metadata;
}

inline std::string OptionalMetadataValue(
    const std::unordered_map<std::string, std::string> &metadata,
    const std::string &key) {
    const auto it = metadata.find(key);
    if (it == metadata.end()) {
        return "";
    }
    return it->second;
}

inline std::string FileStem(const std::string &path) {
    return std::filesystem::path(path).stem().string();
}

inline TypedPackagePaths
ResolveTypedPackagePaths(const std::string &metadata_path,
                         const std::string &expected_algorithm_name) {
    const std::string metadata_name = FileStem(metadata_path);
    if (metadata_name != expected_algorithm_name) {
        throw std::runtime_error("Metadata basename mismatch: expected " +
                                 expected_algorithm_name + ", got " +
                                 metadata_name);
    }

    const auto metadata = LoadMetadataMap(metadata_path);

    return {
        OptionalMetadataValue(metadata, "cpu_f32"),
        OptionalMetadataValue(metadata, "cpu_f64"),
        OptionalMetadataValue(metadata, "cuda_f32"),
        OptionalMetadataValue(metadata, "cuda_f64"),
    };
}

} // namespace algorithm::detail
