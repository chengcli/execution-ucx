/*Copyright 2026 Cheng Li <lumos.leee@gmail.com>. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/ATen.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "axon/c10d/axon_backend.hpp"
#include "axon/c10d/axon_ucx_transport.hpp"

namespace {

constexpr int kWorldSize = 2;
constexpr std::chrono::seconds kTimeout{30};

struct Arguments {
  int rank;
  std::string host;
  uint16_t port;
};

Arguments ParseArguments(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    throw std::runtime_error(
      "usage: process_group_cpu <rank> [store-host] [store-port]");
  }
  return {
    .rank = std::stoi(argv[1]),
    .host = argc >= 3 ? argv[2] : "127.0.0.1",
    .port = static_cast<uint16_t>(argc >= 4 ? std::stoi(argv[3]) : 29500)};
}

::c10::intrusive_ptr<::c10d::ProcessGroup> CreateProcessGroup(
  const Arguments& arguments) {
  ::c10d::TCPStoreOptions store_options;
  store_options.port = arguments.port;
  store_options.isServer = arguments.rank == 0;
  store_options.numWorkers = kWorldSize;
  store_options.timeout =
    std::chrono::duration_cast<std::chrono::milliseconds>(kTimeout);

  auto store =
    ::c10::make_intrusive<::c10d::TCPStore>(arguments.host, store_options);
  auto process_group = ::c10::make_intrusive<::c10d::ProcessGroup>(
    store, arguments.rank, kWorldSize);
  auto transport = std::make_shared<eux::axon::c10d::AxonUcxTransport>(
    store, arguments.rank, kWorldSize, store_options.timeout);
  eux::axon::c10d::InstallAxonBackend(
    process_group, store, std::move(transport), {::c10::DeviceType::CPU},
    store_options.timeout);
  return process_group;
}

void Check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const auto arguments = ParseArguments(argc, argv);
  Check(
    arguments.rank >= 0 && arguments.rank < kWorldSize,
    "rank must be 0 or 1");
  auto process_group = CreateProcessGroup(arguments);

  auto value = at::full({1024}, static_cast<float>(arguments.rank + 1));
  std::vector<at::Tensor> tensors{value};
  process_group->allreduce(tensors)->wait();
  Check(at::equal(value, at::full_like(value, 3)), "CPU allreduce failed");

  auto message = at::full({1024}, -1, at::TensorOptions().dtype(at::kLong));
  std::vector<at::Tensor> messages{message};
  if (arguments.rank == 0) {
    message.fill_(1234);
    process_group->send(messages, 1, 42)->wait();
  } else {
    process_group->recv(messages, 0, 42)->wait();
    Check(
      at::equal(message, at::full_like(message, 1234)), "CPU receive failed");
  }

  process_group->barrier()->wait();
  std::cout << "Rank " << arguments.rank
            << " completed CPU communication through Axon ProcessGroup.\n";
}
