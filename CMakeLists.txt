cmake_minimum_required(VERSION 3.18)

project(
  execution-ucx
  VERSION 0.0.1
  DESCRIPTION "Asynchronous OpenUCX runtime based on the sender/receiver model"
  LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(EXECUTION_UCX_IS_TOP_LEVEL ON)
  include(CTest)
else()
  set(EXECUTION_UCX_IS_TOP_LEVEL OFF)
endif()

option(EXECUTION_UCX_BUILD_UCX "Build the OpenUCX runtime components" OFF)
option(EXECUTION_UCX_BUILD_EXAMPLES "Build examples (requires the UCX components)" OFF)
option(EXECUTION_UCX_ENABLE_CUDA "Enable CUDA support" OFF)
option(EXECUTION_UCX_INSTALL "Generate execution-ucx install rules" ${EXECUTION_UCX_IS_TOP_LEVEL})

add_library(execution_ucx_memory_resource ucx_context/ucx_memory_resource.cpp)
add_library(execution-ucx::memory_resource ALIAS execution_ucx_memory_resource)

target_compile_features(execution_ucx_memory_resource PUBLIC cxx_std_17)
target_include_directories(
  execution_ucx_memory_resource
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set_target_properties(
  execution_ucx_memory_resource
  PROPERTIES
    EXPORT_NAME memory_resource
    POSITION_INDEPENDENT_CODE ON)

if(EXECUTION_UCX_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(execution_ucx_cuda INTERFACE)
  add_library(execution-ucx::cuda ALIAS execution_ucx_cuda)
  target_compile_definitions(execution_ucx_cuda INTERFACE CUDA_ENABLED=1)
  target_include_directories(
    execution_ucx_cuda
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(execution_ucx_cuda INTERFACE CUDA::cudart CUDA::cuda_driver)
  set_target_properties(execution_ucx_cuda PROPERTIES EXPORT_NAME cuda)

  if(BUILD_TESTING)
    add_executable(execution_ucx_cuda_two_gpu_test test_code/cuda_two_gpu_test.cpp)
    target_compile_features(execution_ucx_cuda_two_gpu_test PRIVATE cxx_std_17)
    target_link_libraries(execution_ucx_cuda_two_gpu_test PRIVATE execution-ucx::cuda)

    add_test(NAME execution_ucx.cuda_two_gpu COMMAND execution_ucx_cuda_two_gpu_test)
    set_tests_properties(
      execution_ucx.cuda_two_gpu
      PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 60)
  endif()
endif()

if(EXECUTION_UCX_BUILD_UCX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  find_package(Threads REQUIRED)
  if(TARGET ucx::ucp AND NOT TARGET UCX::ucp)
    add_library(UCX::ucp ALIAS ucx::ucp)
  endif()
  if(NOT TARGET UCX::ucp)
    find_package(UCX 1.18.1 REQUIRED)
  endif()
  if(NOT TARGET unifex::unifex AND NOT TARGET unifex)
    find_package(unifex CONFIG REQUIRED)
  endif()

  if(TARGET unifex AND NOT TARGET unifex::unifex)
    add_library(unifex::unifex ALIAS unifex)
  endif()
  if(NOT TARGET unifex::unifex)
    message(FATAL_ERROR "The libunifex package did not define unifex::unifex or unifex")
  endif()

  add_library(
    execution_ucx_runtime
    ucx_context/ucx_connection.cpp
    ucx_context/ucx_connection_manager.cpp
    ucx_context/ucx_am_context/ucx_am_context.cpp)
  add_library(execution-ucx::runtime ALIAS execution_ucx_runtime)

  target_compile_features(execution_ucx_runtime PUBLIC cxx_std_20)
  target_include_directories(
    execution_ucx_runtime
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(
    execution_ucx_runtime
    PUBLIC
      execution-ucx::memory_resource
      UCX::ucp
      unifex::unifex
      Threads::Threads)
  if(EXECUTION_UCX_ENABLE_CUDA)
    target_link_libraries(execution_ucx_runtime PUBLIC execution-ucx::cuda)
  endif()

  if(CMAKE_DL_LIBS)
    target_link_libraries(execution_ucx_runtime PRIVATE ${CMAKE_DL_LIBS})
  endif()
  if(UNIX AND NOT APPLE)
    target_link_libraries(execution_ucx_runtime PRIVATE rt)
  endif()

  set_target_properties(
    execution_ucx_runtime
    PROPERTIES
      EXPORT_NAME runtime
      POSITION_INDEPENDENT_CODE ON)

  if(EXECUTION_UCX_BUILD_EXAMPLES)
    add_executable(execution_ucx_readme_example test_code/readme.cpp)
    target_compile_features(execution_ucx_readme_example PRIVATE cxx_std_20)
    target_link_libraries(execution_ucx_readme_example PRIVATE execution-ucx::runtime)

    if(EXECUTION_UCX_ENABLE_CUDA)
      add_executable(execution_ucx_cuda_connection_example test_code/test_cuda_connection.cpp)
      target_compile_features(execution_ucx_cuda_connection_example PRIVATE cxx_std_20)
      target_link_libraries(execution_ucx_cuda_connection_example PRIVATE execution-ucx::runtime)

      add_executable(execution_ucx_cuda_single_gpu_example test_code/cuda_single_gpu.cpp)
      target_compile_features(execution_ucx_cuda_single_gpu_example PRIVATE cxx_std_20)
      target_link_libraries(execution_ucx_cuda_single_gpu_example PRIVATE execution-ucx::runtime)
    endif()
  endif()
elseif(EXECUTION_UCX_BUILD_EXAMPLES)
  message(FATAL_ERROR "EXECUTION_UCX_BUILD_EXAMPLES requires EXECUTION_UCX_BUILD_UCX=ON")
endif()

if(EXECUTION_UCX_INSTALL)
  install(
    TARGETS execution_ucx_memory_resource
    EXPORT execution-ucx-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  if(TARGET execution_ucx_runtime)
    install(
      TARGETS execution_ucx_runtime
      EXPORT execution-ucx-targets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  if(TARGET execution_ucx_cuda)
    install(
      TARGETS execution_ucx_cuda
      EXPORT execution-ucx-targets)
  endif()

  install(
    DIRECTORY ucx_context/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ucx_context
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp")

  install(
    EXPORT execution-ucx-targets
    FILE execution-ucx-targets.cmake
    NAMESPACE execution-ucx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/execution-ucx)

  set(EXECUTION_UCX_CONFIG_HAS_RUNTIME "${EXECUTION_UCX_BUILD_UCX}")
  set(EXECUTION_UCX_CONFIG_HAS_CUDA "${EXECUTION_UCX_ENABLE_CUDA}")
  configure_package_config_file(
    cmake/execution-ucx-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/execution-ucx-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/execution-ucx)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/execution-ucx-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/execution-ucx-config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/execution-ucx-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/execution-ucx)

  if(EXECUTION_UCX_BUILD_UCX)
    install(FILES cmake/FindUCX.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/execution-ucx)
  endif()
endif()
