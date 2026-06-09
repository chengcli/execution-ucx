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

#include <utility>

#include <c10/util/Exception.h>

namespace eux::axon::c10d {

AxonBackend::AxonBackend(
  const ::c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
  const ::c10::intrusive_ptr<Options>& options)
  : ::c10d::Backend(rank, size), store_(store), options_(options) {
  TORCH_CHECK(store_.defined(), "AxonBackend requires a c10d Store");
  TORCH_CHECK(options_.defined(), "AxonBackend requires options");
  ValidateTransport();
}

const std::string AxonBackend::getBackendName() const { return "axon"; }

::c10::intrusive_ptr<::c10d::Backend::Options>
AxonBackend::getBackendOptions() {
  return options_;
}

void AxonBackend::setTimeout(std::chrono::milliseconds timeout) {
  options_->timeout = timeout;
}

::c10::intrusive_ptr<::c10d::Work> AxonBackend::send(
  std::vector<at::Tensor>& tensors, int dst_rank, int tag) {
  ValidateTransport();
  TORCH_CHECK(
    dst_rank >= 0 && dst_rank < getSize(), "AxonBackend invalid destination ",
    dst_rank, " for world size ", getSize());
  return options_->transport->Send(tensors, dst_rank, tag);
}

::c10::intrusive_ptr<::c10d::Work> AxonBackend::recv(
  std::vector<at::Tensor>& tensors, int src_rank, int tag) {
  ValidateTransport();
  TORCH_CHECK(
    src_rank >= 0 && src_rank < getSize(), "AxonBackend invalid source ",
    src_rank, " for world size ", getSize());
  return options_->transport->Recv(tensors, src_rank, tag);
}

::c10::intrusive_ptr<::c10d::Work> AxonBackend::barrier(
  const ::c10d::BarrierOptions& options) {
  ValidateTransport();
  return options_->transport->Barrier(options);
}

::c10::intrusive_ptr<::c10d::Work> AxonBackend::allreduce(
  std::vector<at::Tensor>& tensors, const ::c10d::AllreduceOptions& options) {
  ValidateTransport();
  return options_->transport->Allreduce(tensors, options);
}

::c10::intrusive_ptr<::c10d::Work> AxonBackend::reduce(
  std::vector<at::Tensor>& tensors, const ::c10d::ReduceOptions& options) {
  ValidateTransport();
  return options_->transport->Reduce(tensors, options);
}

void AxonBackend::ValidateTransport() const {
  TORCH_CHECK(
    options_.defined() && options_->transport,
    "AxonBackend requires a configured AxonTransport");
}

::c10::intrusive_ptr<AxonBackend> InstallAxonBackend(
  const ::c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
  const ::c10::intrusive_ptr<::c10d::Store>& store,
  std::shared_ptr<AxonTransport> transport,
  std::vector<::c10::DeviceType> device_types,
  std::chrono::milliseconds timeout) {
  TORCH_CHECK(
    process_group.defined(), "InstallAxonBackend requires a ProcessGroup");
  TORCH_CHECK(
    !device_types.empty(), "InstallAxonBackend requires a device type");

  auto options =
    ::c10::make_intrusive<AxonBackend::Options>(std::move(transport), timeout);
  auto backend = ::c10::make_intrusive<AxonBackend>(
    store, process_group->getRank(), process_group->getSize(), options);

  process_group->setDefaultBackend("axon");
  for (const auto device_type : device_types) {
    process_group->setBackend(
      device_type, ::c10d::ProcessGroup::BackendType::CUSTOM, backend);
  }
  return backend;
}

}  // namespace eux::axon::c10d
