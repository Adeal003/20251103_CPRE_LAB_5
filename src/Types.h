#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

namespace ML {

// --- Data Types ---
using size = std::size_t;
using dimVec = std::vector<std::size_t>;

// Signed int
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// Unsigned int
using ui8 = std::uint8_t;
using ui16 = std::uint16_t;
using ui32 = std::uint32_t;
using ui64 = std::uint64_t;

// Floating point
using fp32 = float;
using fp64 = double;
using fp96 = long double;

}  // namespace ML