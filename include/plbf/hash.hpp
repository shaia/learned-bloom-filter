#pragma once
//
// 64-bit key hashes for the backup filters.
//
// The filters consume the hash in two halves -- the high 32 bits pick a block,
// the low 32 bits pick the bits within it -- so both halves need to be well
// mixed. Anything with a full 64-bit avalanche will do; these two are here so
// the library has no dependency, not because they are special.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace plbf {

// The splitmix64 finaliser: a bijective 64-bit mix, for integer keys.
constexpr std::uint64_t hash_u64(std::uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// MurmurHash64A (Austin Appleby, public domain), for byte-string keys.
inline std::uint64_t hash_bytes(const void* data, std::size_t len, std::uint64_t seed = 0) noexcept {
    constexpr std::uint64_t m = 0xc6a4a7935bd1e995ULL;
    constexpr int r = 47;
    std::uint64_t h = seed ^ (static_cast<std::uint64_t>(len) * m);

    const auto* p = static_cast<const unsigned char*>(data);
    const unsigned char* end = p + (len & ~std::size_t{7});
    for (; p != end; p += 8) {
        std::uint64_t k;
        std::memcpy(&k, p, 8);  // unaligned-safe; compiles to one load
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }
    switch (len & 7) {
        case 7: h ^= std::uint64_t{p[6]} << 48; [[fallthrough]];
        case 6: h ^= std::uint64_t{p[5]} << 40; [[fallthrough]];
        case 5: h ^= std::uint64_t{p[4]} << 32; [[fallthrough]];
        case 4: h ^= std::uint64_t{p[3]} << 24; [[fallthrough]];
        case 3: h ^= std::uint64_t{p[2]} << 16; [[fallthrough]];
        case 2: h ^= std::uint64_t{p[1]} << 8; [[fallthrough]];
        case 1:
            h ^= std::uint64_t{p[0]};
            h *= m;
            break;
        default: break;
    }
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

inline std::uint64_t hash_bytes(std::string_view s, std::uint64_t seed = 0) noexcept {
    return hash_bytes(s.data(), s.size(), seed);
}

}  // namespace plbf
