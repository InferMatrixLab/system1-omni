#pragma once

#include <cstddef>
#include <string_view>

namespace qwen3_5 {
// Empirical numerical evidence covers this reported device profile only.
// Matching SM90 alone is insufficient: do not enable on H100/H200 by inference.
// Unknown names, partition sizes and library versions keep exact algo matching.
constexpr bool equivalent_tiles_device(std::string_view name, int major, int minor,
                                       int multiprocessors, std::size_t lt_version) {
    return name == "NVIDIA L20X" && major == 9 && minor == 0 &&
           multiprocessors == 132 && lt_version == 130101;
}
}  // namespace qwen3_5
