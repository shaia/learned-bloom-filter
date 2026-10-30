#pragma once
//
// Partitioned learned Bloom filter: header-only C++20 port of the construction
// methods in Sato and Matsui, "Fast Construction of Partitioned Learned Bloom
// Filter with Theoretical Guarantees" (The VLDB Journal, 2026), with
// split-block backup filters and a SIMD batch query.

#include "filter.hpp"
#include "hash.hpp"
#include "optimize.hpp"
#include "region_tables.hpp"
#include "row_maxima.hpp"
#include "segments.hpp"
#include "simd.hpp"
#include "split_block.hpp"
