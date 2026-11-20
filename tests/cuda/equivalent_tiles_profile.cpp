#include "src/backends/cuda/qwen3_5/gemm_profile.h"

using qwen3_5::equivalent_tiles_device;
static_assert(equivalent_tiles_device("NVIDIA L20X", 9, 0, 132, 130101));
static_assert(!equivalent_tiles_device("NVIDIA H100 80GB HBM3", 9, 0, 132, 130101));
static_assert(!equivalent_tiles_device("NVIDIA H200", 9, 0, 132, 130101));
static_assert(!equivalent_tiles_device("", 9, 0, 132, 130101));
static_assert(!equivalent_tiles_device("NVIDIA L20X", 8, 9, 132, 130101));
static_assert(!equivalent_tiles_device("NVIDIA L20X", 9, 1, 132, 130101));
static_assert(!equivalent_tiles_device("NVIDIA L20X", 9, 0, 66, 130101));
static_assert(!equivalent_tiles_device("NVIDIA L20X", 9, 0, 132, 130100));
static_assert(!equivalent_tiles_device("NVIDIA L20X", 9, 0, 132, 130102));
int main() { return 0; }
