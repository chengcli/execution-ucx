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

#include "axon/c10d/axon_backend.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <torch/csrc/distributed/c10d/HashStore.hpp>

namespace {

class CompletedWork final : public c10d::Work {
 public:
  explicit CompletedWork(c10d::OpType type) : c10d::Work(0, type) { finish(); }
};

class GpuLoopbackTransport final : public eux::axon::c10d::AxonTransport {
 public:
  c10::intrusive_ptr<c10d::Work> Send(
    std::vector<at::Tensor>& tensors, int, int tag) override {
    ValidateSingleCudaTensor(tensors);
    messages_.emplace_back(tag, tensors.front().to(at::kCPU));
    return c10::make_intrusive<CompletedWork>(c10d::OpType::SEND);
  }

  c10::intrusive_ptr<c10d::Work> Recv(
    std::vector<at::Tensor>& tensors, int, int tag) override {
    ValidateSingleCudaTensor(tensors);
    if (messages_.empty() || messages_.front().first != tag) {
      throw std::runtime_error("GPU loopback message tag mismatch");
    }
    tensors.front().copy_(messages_.front().second);
    messages_.pop_front();
    return c10::make_intrusive<CompletedWork>(c10d::OpType::RECV);
  }

  c10::intrusive_ptr<c10d::Work> Barrier(const c10d::BarrierOptions&) override {
    return c10::make_intrusive<CompletedWork>(c10d::OpType::BARRIER);
  }

  c10::intrusive_ptr<c10d::Work> Allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions&) override {
    ValidateSingleCudaTensor(tensors);
    tensors.front().mul_(2);
    return c10::make_intrusive<CompletedWork>(c10d::OpType::ALLREDUCE);
  }

  c10::intrusive_ptr<c10d::Work> Reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions&) override {
    ValidateSingleCudaTensor(tensors);
    tensors.front().mul_(2);
    return c10::make_intrusive<CompletedWork>(c10d::OpType::REDUCE);
  }

 private:
  static void ValidateSingleCudaTensor(const std::vector<at::Tensor>& tensors) {
    if (tensors.size() != 1 || !tensors.front().is_cuda()) {
      throw std::runtime_error("GPU loopback requires one CUDA tensor");
    }
  }

  std::deque<std::pair<int, at::Tensor>> messages_;
};

}  // namespace

int main() {
  const int device_count = at::cuda::device_count();
  if (device_count < 2) {
    std::cout << "Axon c10d GPU test skipped: found " << device_count
              << " GPU(s).\n";
    return 77;
  }

  auto store = c10::make_intrusive<c10d::HashStore>();
  auto process_group = c10::make_intrusive<c10d::ProcessGroup>(store, 0, 2);
  auto transport = std::make_shared<GpuLoopbackTransport>();
  eux::axon::c10d::InstallAxonBackend(
    process_group, store, transport, {c10::DeviceType::CUDA});

  constexpr int64_t element_count = 1024 * 1024;
  constexpr int tag = 42;
  auto source = at::arange(
    element_count, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto destination = at::zeros(
    {element_count},
    at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 1));

  std::vector<at::Tensor> send_tensors{source};
  std::vector<at::Tensor> recv_tensors{destination};
  process_group->send(send_tensors, 1, tag)->wait();
  process_group->recv(recv_tensors, 1, tag)->wait();

  auto expected = source.to(at::Device(at::kCUDA, 1));
  assert(at::equal(destination, expected));

  process_group->allreduce(recv_tensors)->wait();
  assert(at::equal(destination, expected * 2));

  std::cout << "Validated Axon c10d CUDA routing across GPU 0 and GPU 1.\n";
}
