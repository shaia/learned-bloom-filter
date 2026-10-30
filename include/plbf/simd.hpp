#pragma once
//
// What the AVX2 kernels share: the x86-64 feature macros and the runtime check.

#if defined(__x86_64__) || defined(_M_X64)
#define PLBF_X86_64 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#define PLBF_X86_64 0
#endif

// GCC and Clang only emit AVX2 inside functions marked for it unless the whole
// TU is built with -mavx2; marking just the kernels keeps the rest of the
// library -- the optimiser in particular -- free of FMA contraction, which
// would cost bit-exact parity with the reference.
#if PLBF_X86_64 && (defined(__GNUC__) || defined(__clang__))
#define PLBF_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define PLBF_TARGET_AVX2
#endif

namespace plbf {

// Whether this CPU and OS can run the AVX2 kernels. Not cached: callers that
// care (filter does) ask once and keep the answer, and a function-local static
// would be a hidden shared write.
inline bool cpu_has_avx2() noexcept {
#if !PLBF_X86_64
    return false;
#elif defined(_MSC_VER)
    int info[4];
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    const bool avx = (info[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // OS saves XMM and YMM state
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2") != 0;
#endif
}

}  // namespace plbf
