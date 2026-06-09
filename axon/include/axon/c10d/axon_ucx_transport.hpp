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

#pragma once

#ifndef AXON_C10D_AXON_UCX_TRANSPORT_HPP_
#define AXON_C10D_AXON_UCX_TRANSPORT_HPP_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include <ucp/api/ucp.h>

#include "axon/c10d/axon_backend.hpp"

namespace eux::axon::c10d {

class AxonUcxTransport final : public AxonTransport {
 public:
  AxonUcxTransport(
    ::c10::intrusive_ptr<::c10d::Store> store, int rank, int world_size,
    std::chrono::milliseconds timeout = kBackendDefaultTimeout);
  ~AxonUcxTransport() override;

  ::c10::intrusive_ptr<::c10d::Work> Send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) override;
  ::c10::intrusive_ptr<::c10d::Work> Recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) override;
  ::c10::intrusive_ptr<::c10d::Work> Barrier(
    const ::c10d::BarrierOptions& options) override;
  ::c10::intrusive_ptr<::c10d::Work> Allreduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::AllreduceOptions& options) override;
  ::c10::intrusive_ptr<::c10d::Work> Reduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::ReduceOptions& options) override;

 private:
  static constexpr uint64_t kPointToPointKind = 1;
  static constexpr uint64_t kBarrierKind = 2;
  static constexpr uint64_t kReduceKind = 3;
  static constexpr uint64_t kBroadcastKind = 4;

  ::c10::intrusive_ptr<::c10d::Work> Run(
    ::c10d::OpType type, const std::function<void()>& operation);
  void SendTensors(
    const std::vector<at::Tensor>& tensors, int dst_rank, uint64_t tag);
  void RecvTensors(
    std::vector<at::Tensor>& tensors, int src_rank, uint64_t tag);
  void SynchronizeTensors(const std::vector<at::Tensor>& tensors) const;
  void WaitRequest(ucs_status_ptr_t request);
  void ValidateTensors(const std::vector<at::Tensor>& tensors) const;
  void ValidateSum(const ::c10d::ReduceOp& op) const;
  uint64_t PointToPointTag(int src_rank, int dst_rank, int tag) const;
  uint64_t CollectiveTag(uint64_t kind, uint64_t sequence, int rank) const;

  ::c10::intrusive_ptr<::c10d::Store> store_;
  int rank_;
  int world_size_;
  std::chrono::milliseconds timeout_;
  ucp_context_h context_ = nullptr;
  ucp_worker_h worker_ = nullptr;
  std::vector<ucp_ep_h> endpoints_;
  std::atomic<uint64_t> collective_sequence_{0};
  std::mutex mutex_;
};

}  // namespace eux::axon::c10d

#endif  // AXON_C10D_AXON_UCX_TRANSPORT_HPP_
