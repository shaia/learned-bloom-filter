# One binary per suite: a failure then tells you which area broke without
# bisecting a single monolithic test.

set(PLBF_TESTS
  test_optimize
  test_filter)

foreach(name IN LISTS PLBF_TESTS)
  add_executable(${name} ${name}.cpp)
  set_target_properties(${name} PROPERTIES OUTPUT_NAME ${name}${PLBF_BIN_SUFFIX})
  target_link_libraries(${name} PRIVATE plbf::plbf plbf_dev_flags)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE PLBF_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 600)
endforeach()

if(PLBF_BUILD_CUDA)
  add_executable(test_cuda test_cuda.cu)
  set_target_properties(test_cuda PROPERTIES OUTPUT_NAME test_cuda${PLBF_BIN_SUFFIX})
  target_link_libraries(test_cuda PRIVATE plbf::cuda plbf_dev_flags)
  target_include_directories(test_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_cuda COMMAND test_cuda)
  set_tests_properties(test_cuda PROPERTIES TIMEOUT 600)
endif()
