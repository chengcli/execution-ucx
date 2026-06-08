cmake_minimum_required(VERSION 3.24)

project(
  execution-ucx
  VERSION 0.0.1
  DESCRIPTION "Asynchronous OpenUCX runtime based on the sender/receiver model"
  LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(EXECUTION_UCX_BUILD_UCX "Build the OpenUCX runtime components" OFF)
option(EXECUTION_UCX_BUILD_EXAMPLES "Build examples (requires the UCX components)" OFF)

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

if(EXECUTION_UCX_BUILD_UCX)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  find_package(Threads REQUIRED)
  find_package(UCX 1.18.1 REQUIRED)
  find_package(unifex CONFIG REQUIRED)

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
  endif()
elseif(EXECUTION_UCX_BUILD_EXAMPLES)
  message(FATAL_ERROR "EXECUTION_UCX_BUILD_EXAMPLES requires EXECUTION_UCX_BUILD_UCX=ON")
endif()

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
