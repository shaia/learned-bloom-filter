cmake_minimum_required(VERSION 3.20)

# Default to Release, decided before project(): CMake's MSVC platform file sets
# CMAKE_BUILD_TYPE_INIT to Debug, so the usual "if(NOT CMAKE_BUILD_TYPE)" after
# project() never fires there, and an MSVC build ends up silently Debug -- 3 to
# 24x slower here, and with the kernels in a different order
# (bench/results/2026-10-30/msvc_debug_vs_release.txt).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type for single-config generators")
endif()

project(plbf
  VERSION 0.1.0
  DESCRIPTION "Header-only C++20 partitioned learned Bloom filter: fast construction, SIMD queries"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # so clangd sees the right flags

if(PROJECT_IS_TOP_LEVEL)
  # Binaries land in <project>/bin regardless of which build directory
  # produced them, so the thing you run is in a predictable place.
  set(PLBF_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PLBF_BIN_DIR})
  foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${cfg} cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} ${PLBF_BIN_DIR})
  endforeach()
endif()

option(PLBF_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(PLBF_BUILD_BENCH "Build the benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(PLBF_BUILD_CUDA "Build the CUDA tests and benchmarks (needs nvcc; MSVC host compiler on Windows)" OFF)

# Builds from different toolchains share bin/, and silently overwriting each
# other there is how you end up benchmarking the wrong compiler's binary. The
# CUDA build has to use MSVC on Windows, so MSVC gets a suffix by default.
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(PLBF_DEFAULT_SUFFIX "_msvc")
else()
  set(PLBF_DEFAULT_SUFFIX "")
endif()
set(PLBF_BIN_SUFFIX "${PLBF_DEFAULT_SUFFIX}" CACHE STRING "Appended to every binary's name")

# --- the library -----------------------------------------------------------

add_library(plbf INTERFACE)
add_library(plbf::plbf ALIAS plbf)
target_include_directories(plbf INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(plbf INTERFACE cxx_std_20)

# --- shared settings for our own targets -----------------------------------
#
# INTERFACE-only so nothing here leaks to consumers of the library.
add_library(plbf_dev_flags INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(plbf_dev_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /fp:precise>)
else()
  # No FMA contraction: the golden tests compare against the Python reference
  # to the last bit, and a fused multiply-add rounds once where Python rounds
  # twice.
  target_compile_options(plbf_dev_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -ffp-contract=off>)
endif()

# --- the CUDA backend --------------------------------------------------------
#
# Header-only as well: include/plbf/cuda/filter.cuh, compiled by the
# consumer's nvcc. This only adds the target that carries the runtime library,
# and our own tests and benchmarks.
if(PLBF_BUILD_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)  # the GPU in this machine; set it explicitly to cross-compile
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  add_library(plbf_cuda INTERFACE)
  add_library(plbf::cuda ALIAS plbf_cuda)
  target_link_libraries(plbf_cuda INTERFACE plbf CUDA::cudart)

  # A host function called from device code is a hard error rather than a
  # warning: it means a kernel has drifted from the CPU code it mirrors.
  target_compile_options(plbf_dev_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=cross-execution-space-call>)
  if(MSVC)
    target_compile_options(plbf_dev_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/permissive->)
  endif()
endif()

if(MINGW)
  # A MinGW binary resolves libstdc++-6.dll through PATH, and the first copy
  # found is often another toolchain's (Git for Windows ships one). The ABI
  # mismatch crashes before main with nothing on stderr; linking the runtime
  # in removes the lottery.
  target_link_options(plbf_dev_flags INTERFACE -static)
endif()

# --- tests and benchmarks --------------------------------------------------

if(PLBF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(PLBF_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# --- install ---------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS plbf EXPORT plbf-targets)
install(DIRECTORY include/plbf DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT plbf-targets
  FILE plbf-targets.cmake
  NAMESPACE plbf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plbf)
