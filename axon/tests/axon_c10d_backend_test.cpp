/*Copyright 2026 He Jia <mofhejia@163.com>. All Rights Reserved.

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

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

#include <torch/csrc/distributed/c10d/HashStore.hpp>

namespace {

class CompletedWork final : public c10d::Work {
 public:
  explicit CompletedWork(c10d::OpType type) : c10d::Work(0, type) { finish(); }
};

class RecordingTransport final : public eux::axon::c10d::AxonTransport {
 public:
  c10::intrusive_ptr<c10d::Work> Send(
    std::vector<at::Tensor>&, int dst_rank, int tag) override {
    last_rank = dst_rank;
    last_tag = tag;
    ++send_count;
    return c10::make_intrusive<CompletedWork>(c10d::OpType::SEND);
  }

  c10::intrusive_ptr<c10d::Work> Recv(
    std::vector<at::Tensor>&, int src_rank, int tag) override {
    last_rank = src_rank;
    last_tag = tag;
    ++recv_count;
    return c10::make_intrusive<CompletedWork>(c10d::OpType::RECV);
  }

  c10::intrusive_ptr<c10d::Work> Barrier(const c10d::BarrierOptions&) override {
    ++barrier_count;
    return c10::make_intrusive<CompletedWork>(c10d::OpType::BARRIER);
  }

  c10::intrusive_ptr<c10d::Work> Allreduce(
    std::vector<at::Tensor>&, const c10d::AllreduceOptions&) override {
    ++allreduce_count;
    return c10::make_intrusive<CompletedWork>(c10d::OpType::ALLREDUCE);
  }

  c10::intrusive_ptr<c10d::Work> Reduce(
    std::vector<at::Tensor>&, const c10d::ReduceOptions&) override {
    ++reduce_count;
    return c10::make_intrusive<CompletedWork>(c10d::OpType::REDUCE);
  }

  int send_count = 0;
  int recv_count = 0;
  int barrier_count = 0;
  int allreduce_count = 0;
  int reduce_count = 0;
  int last_rank = -1;
  int last_tag = -1;
};

}  // namespace

int main() {
  auto store = c10::make_intrusive<c10d::HashStore>();
  auto process_group = c10::make_intrusive<c10d::ProcessGroup>(store, 0, 2);
  auto transport = std::make_shared<RecordingTransport>();
  auto backend = eux::axon::c10d::InstallAxonBackend(
    process_group, store, transport, {c10::DeviceType::CPU});

  assert(backend->getBackendName() == "axon");
  assert(process_group->getBackendName() == "custom");

  std::vector<at::Tensor> tensors;
  backend->send(tensors, 1, 11)->wait();
  backend->recv(tensors, 1, 12)->wait();
  backend->barrier()->wait();
  backend->allreduce(tensors)->wait();
  backend->reduce(tensors)->wait();

  assert(transport->send_count == 1);
  assert(transport->recv_count == 1);
  assert(transport->barrier_count == 1);
  assert(transport->allreduce_count == 1);
  assert(transport->reduce_count == 1);
  assert(transport->last_rank == 1);
  assert(transport->last_tag == 12);

  bool rejected = false;
  try {
    backend->send(tensors, 2, 0);
  } catch (const c10::Error&) {
    rejected = true;
  }
  assert(rejected);
}
