#pragma once

#include <cstdint>

// lambda is fixed 32.32
uint32_t poisson_random_variable_fixed_int(uint64_t& seed, int64_t lambda);

