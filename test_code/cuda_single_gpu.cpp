#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unifex/inplace_stop_token.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/task.hpp>

#include "ucx_context/cuda/ucx_cuda_memory_manager.hpp"
#include "ucx_context/ucx_am_context/ucx_am_context.hpp"

namespace {

using eux::ucxx::connect_endpoint;
using eux::ucxx::connection_recv;
using eux::ucxx::connection_send;
using eux::ucxx::UcxAmData;
using eux::ucxx::UcxCudaMemoryResourceManager;
using eux::ucxx::ucx_am_context;
using unifex::task;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

task<UcxAmData> receive_on_gpu(ucx_am_context::scheduler scheduler) {
  auto message = co_await connection_recv(scheduler, ucx_memory_type::CUDA);
  co_return message.move_data();
}

task<void> send_from_gpu(
  ucx_am_context::scheduler scheduler, std::vector<std::byte>& peer_address,
  ucx_am_data_t& message) {
  const auto connection_id =
    co_await connect_endpoint(scheduler, peer_address);
  co_await connection_send(scheduler, connection_id, message);
}

}  // namespace

int main() {
  constexpr int device_id = 0;
  constexpr size_t payload_size = 1024 * 1024;
  constexpr char header[] = "single-gpu-ucx";

  check_cuda(cudaSetDevice(device_id), "cudaSetDevice");

  cudaDeviceProp properties{};
  check_cuda(
    cudaGetDeviceProperties(&properties, device_id),
    "cudaGetDeviceProperties");
  std::cout << "Using GPU 0: " << properties.name << '\n';

  UcxCudaMemoryResourceManager server_memory;
  UcxCudaMemoryResourceManager client_memory;
  ucx_am_context server_context(server_memory, "cuda-single-gpu-server");
  ucx_am_context client_context(client_memory, "cuda-single-gpu-client");

  unifex::inplace_stop_source server_stop;
  unifex::inplace_stop_source client_stop;
  std::thread server_thread([&] {
    check_cuda(cudaSetDevice(device_id), "server cudaSetDevice");
    server_context.run(server_stop.get_token());
  });
  std::thread client_thread([&] {
    check_cuda(cudaSetDevice(device_id), "client cudaSetDevice");
    client_context.run(client_stop.get_token());
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::vector<std::byte> server_address;
  if (auto ec = server_context.get_ucp_address(server_address)) {
    throw std::system_error(ec, "get server UCP address");
  }

  std::vector<std::byte> expected(payload_size);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<std::byte>(i % 251);
  }

  void* send_device = client_memory.allocate(ucx_memory_type::CUDA, payload_size);
  client_memory.memcpy(
    ucx_memory_type::CUDA, send_device, ucx_memory_type::HOST, expected.data(),
    expected.size());

  ucx_am_data_t message{};
  message.header = {
    const_cast<char*>(header),
    sizeof(header),
  };
  message.buffer = {send_device, payload_size};
  message.buffer_type = ucx_memory_type::CUDA;

  auto receive_future = std::async(std::launch::async, [&] {
    return unifex::sync_wait(receive_on_gpu(server_context.get_scheduler()));
  });

  unifex::sync_wait(send_from_gpu(
    client_context.get_scheduler(), server_address, message));

  auto received = receive_future.get();
  if (!received.has_value()) {
    throw std::runtime_error("receive operation stopped before completion");
  }

  std::vector<std::byte> actual(payload_size);
  server_memory.memcpy(
    ucx_memory_type::HOST, actual.data(), ucx_memory_type::CUDA,
    received->get()->buffer.data, received->get()->buffer.size);

  client_memory.deallocate(
    ucx_memory_type::CUDA, send_device, payload_size);

  if (received->get()->buffer.size != expected.size() || actual != expected) {
    throw std::runtime_error("received CUDA payload does not match");
  }
  if (
    received->get()->header.size != sizeof(header)
    || std::memcmp(received->get()->header.data, header, sizeof(header)) != 0) {
    throw std::runtime_error("received header does not match");
  }

  client_stop.request_stop();
  server_stop.request_stop();
  client_thread.join();
  server_thread.join();

  std::cout << "Transferred and validated " << payload_size
            << " bytes through CUDA device memory on GPU 0.\n";
  return 0;
}
