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

#ifndef AXON_C10D_AXON_BACKEND_HPP_
#define AXON_C10D_AXON_BACKEND_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace eux::axon::c10d {

/**
 * Transport contract used by AxonBackend.
 *
 * The contract intentionally matches the subset of c10d operations used by
 * Snapy's ProcessGroup integration. Implementations own rendezvous, progress,
 * tensor memory handling, and c10d::Work completion.
 */
class AxonTransport {
 public:
  virtual ~AxonTransport() = default;

  virtual ::c10::intrusive_ptr<::c10d::Work> Send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) = 0;
  virtual ::c10::intrusive_ptr<::c10d::Work> Recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) = 0;
  virtual ::c10::intrusive_ptr<::c10d::Work> Barrier(
    const ::c10d::BarrierOptions& options) = 0;
  virtual ::c10::intrusive_ptr<::c10d::Work> Allreduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::AllreduceOptions& options) = 0;
  virtual ::c10::intrusive_ptr<::c10d::Work> Reduce(
    std::vector<at::Tensor>& tensors, const ::c10d::ReduceOptions& options) = 0;
};

/**
 * c10d backend adapter for Axon transports.
 *
 * Install this backend on a generic c10d::ProcessGroup in the same way Snapy
 * installs Gloo and NCCL. ProcessGroup remains responsible for selecting the
 * backend by tensor device type.
 */
class AxonBackend final : public ::c10d::Backend {
 public:
  struct Options final : public ::c10d::Backend::Options {
    explicit Options(
      std::shared_ptr<AxonTransport> transport,
      std::chrono::milliseconds timeout = kBackendDefaultTimeout)
      : ::c10d::Backend::Options("axon", timeout),
        transport(std::move(transport)) {}

    std::shared_ptr<AxonTransport> transport;
  };

  AxonBackend(
    const ::c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const ::c10::intrusive_ptr<Options>& options);
  ~AxonBackend() override = default;

  const std::string getBackendName() const override;
  ::c10::intrusive_ptr<::c10d::Backend::Options> getBackendOptions() override;
  void setTimeout(std::chrono::milliseconds timeout) override;

  ::c10::intrusive_ptr<::c10d::Work> send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) override;
  ::c10::intrusive_ptr<::c10d::Work> recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) override;
  ::c10::intrusive_ptr<::c10d::Work> barrier(
    const ::c10d::BarrierOptions& options = {}) override;
  ::c10::intrusive_ptr<::c10d::Work> allreduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::AllreduceOptions& options = {}) override;
  ::c10::intrusive_ptr<::c10d::Work> reduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::ReduceOptions& options = {}) override;

 private:
  void ValidateTransport() const;

  ::c10::intrusive_ptr<::c10d::Store> store_;
  ::c10::intrusive_ptr<Options> options_;
};

/**
 * Create and install an Axon backend on a generic ProcessGroup.
 *
 * A single backend instance is shared by every listed device type, matching
 * ProcessGroup's handling of custom backends.
 */
::c10::intrusive_ptr<AxonBackend> InstallAxonBackend(
  const ::c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
  const ::c10::intrusive_ptr<::c10d::Store>& store,
  std::shared_ptr<AxonTransport> transport,
  std::vector<::c10::DeviceType> device_types = {::c10::DeviceType::CPU},
  std::chrono::milliseconds timeout = kBackendDefaultTimeout);

}  // namespace eux::axon::c10d

#endif  // AXON_C10D_AXON_BACKEND_HPP_
