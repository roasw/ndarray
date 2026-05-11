#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace algorithm::detail {

struct TypedPackagePaths {
    std::string float_path;
    std::string double_path;
};

inline std::unordered_map<std::string, std::string>
LoadPackageMap(const std::string &metadata_path) {
    std::ifstream in(metadata_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open AOTI metadata file: " +
                                 metadata_path);
    }

    std::unordered_map<std::string, std::string> package_paths;
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
        static constexpr const char *kPackagePrefix = "package:";
        if (key.rfind(kPackagePrefix, 0) == 0) {
            const std::string package_name = key.substr(8);
            if (package_name.empty() || value.empty()) {
                throw std::runtime_error(
                    "Invalid package entry in metadata file: " + metadata_path);
            }
            package_paths[package_name] = value;
        }
    }

    if (package_paths.empty()) {
        throw std::runtime_error("No package entries found in metadata file: " +
                                 metadata_path);
    }

    return package_paths;
}

inline std::string
RequirePackagePath(const std::unordered_map<std::string, std::string> &packages,
                   const std::string &package_name) {
    const auto it = packages.find(package_name);
    if (it == packages.end()) {
        throw std::runtime_error("Package not found in metadata: " +
                                 package_name);
    }
    return it->second;
}

inline TypedPackagePaths
ResolveTypedPackagePaths(const std::string &metadata_path,
                         const std::string &f32_package_name,
                         const std::string &f64_package_name) {
    const auto packages = LoadPackageMap(metadata_path);
    return {
        RequirePackagePath(packages, f32_package_name),
        RequirePackagePath(packages, f64_package_name),
    };
}

} // namespace algorithm::detail
