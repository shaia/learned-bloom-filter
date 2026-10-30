cmake_minimum_required(VERSION 3.20)
project(default_type CXX)
# The usual guard: it never fires under MSVC, because the platform already set Debug.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  message(STATUS "guard fired: defaulting to Release")
endif()
message(STATUS "CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}', flags '${CMAKE_CXX_FLAGS_DEBUG}'")
