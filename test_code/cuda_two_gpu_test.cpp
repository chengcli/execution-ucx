#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_ENABLED
#error "execution-ucx::cuda must define CUDA_ENABLED"
#endif

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void verify_device_pointer(const void* pointer, int expected_device) {
  cudaPointerAttributes attributes{};
  check_cuda(
    cudaPointerGetAttributes(&attributes, pointer), "cudaPointerGetAttributes");
  if (
    attributes.type != cudaMemoryTypeDevice
    || attributes.device != expected_device) {
    throw std::runtime_error("CUDA allocation belongs to the wrong GPU");
  }
}

}  // namespace

int main() {
  constexpr int source_device = 0;
  constexpr int destination_device = 1;
  constexpr size_t payload_size = 1024 * 1024;

  int device_count = 0;
  const auto device_count_status = cudaGetDeviceCount(&device_count);
  if (device_count_status == cudaErrorNoDevice) {
    std::cout << "Two-GPU test skipped: no CUDA GPUs are visible.\n";
    return 77;
  }
  check_cuda(device_count_status, "cudaGetDeviceCount");
  if (device_count < 2) {
    std::cout << "Two-GPU test skipped: found " << device_count << " GPU(s).\n";
    return 77;
  }

  std::vector<std::byte> expected(payload_size);
  std::vector<std::byte> actual(payload_size);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<std::byte>((i * 17 + 3) % 251);
  }

  void* source = nullptr;
  void* destination = nullptr;

  check_cuda(cudaSetDevice(source_device), "cudaSetDevice(source)");
  check_cuda(cudaMalloc(&source, payload_size), "cudaMalloc(source)");
  verify_device_pointer(source, source_device);
  check_cuda(
    cudaMemcpy(source, expected.data(), payload_size, cudaMemcpyHostToDevice),
    "cudaMemcpy(source host-to-device)");

  check_cuda(cudaSetDevice(destination_device), "cudaSetDevice(destination)");
  check_cuda(cudaMalloc(&destination, payload_size), "cudaMalloc(destination)");
  verify_device_pointer(destination, destination_device);

  int can_access_peer = 0;
  check_cuda(
    cudaDeviceCanAccessPeer(
      &can_access_peer, destination_device, source_device),
    "cudaDeviceCanAccessPeer");

  if (can_access_peer) {
    const auto enable_status = cudaDeviceEnablePeerAccess(source_device, 0);
    if (
      enable_status != cudaSuccess
      && enable_status != cudaErrorPeerAccessAlreadyEnabled) {
      check_cuda(enable_status, "cudaDeviceEnablePeerAccess");
    }
    if (enable_status == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
    }

    check_cuda(
      cudaMemcpyPeer(
        destination, destination_device, source, source_device, payload_size),
      "cudaMemcpyPeer");
    std::cout << "Transferred data using CUDA peer access.\n";
  } else {
    std::vector<std::byte> staging(payload_size);
    check_cuda(cudaSetDevice(source_device), "cudaSetDevice(source staging)");
    check_cuda(
      cudaMemcpy(staging.data(), source, payload_size, cudaMemcpyDeviceToHost),
      "cudaMemcpy(source device-to-host)");
    check_cuda(
      cudaSetDevice(destination_device), "cudaSetDevice(destination staging)");
    check_cuda(
      cudaMemcpy(
        destination, staging.data(), payload_size, cudaMemcpyHostToDevice),
      "cudaMemcpy(destination host-to-device)");
    std::cout
      << "CUDA peer access unavailable; transferred through host memory.\n";
  }

  check_cuda(
    cudaMemcpy(
      actual.data(), destination, payload_size, cudaMemcpyDeviceToHost),
    "cudaMemcpy(destination device-to-host)");

  check_cuda(cudaFree(destination), "cudaFree(destination)");
  check_cuda(cudaSetDevice(source_device), "cudaSetDevice(source cleanup)");
  check_cuda(cudaFree(source), "cudaFree(source)");

  if (actual != expected) {
    throw std::runtime_error("two-GPU CUDA payload does not match");
  }

  std::cout << "Validated " << payload_size
            << " bytes across GPU 0 and GPU 1.\n";
  return 0;
}
