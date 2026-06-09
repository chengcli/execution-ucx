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

#ifdef AXON_C10D_UCX_ENABLED
#include "axon/c10d/axon_ucx_transport.hpp"
#endif

#include <chrono>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace eux::axon::python {

class __attribute__((visibility("hidden"))) PythonAxonTransport final
  : public c10d::AxonTransport {
 public:
  explicit PythonAxonTransport(py::object transport)
    : transport_(std::move(transport)) {}

  ~PythonAxonTransport() override {
    py::gil_scoped_acquire gil;
    transport_.reset();
  }

  ::c10::intrusive_ptr<::c10d::Work> Send(
    std::vector<at::Tensor>& tensors, int dst_rank, int tag) override {
    return CallWork("send", tensors, dst_rank, tag);
  }

  ::c10::intrusive_ptr<::c10d::Work> Recv(
    std::vector<at::Tensor>& tensors, int src_rank, int tag) override {
    return CallWork("recv", tensors, src_rank, tag);
  }

  ::c10::intrusive_ptr<::c10d::Work> Barrier(
    const ::c10d::BarrierOptions& options) override {
    py::gil_scoped_acquire gil;
    return transport_->attr("barrier")(options)
      .cast<::c10::intrusive_ptr<::c10d::Work>>();
  }

  ::c10::intrusive_ptr<::c10d::Work> Allreduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::AllreduceOptions& options) override {
    return CallWork("allreduce", tensors, options);
  }

  ::c10::intrusive_ptr<::c10d::Work> Reduce(
    std::vector<at::Tensor>& tensors,
    const ::c10d::ReduceOptions& options) override {
    return CallWork("reduce", tensors, options);
  }

 private:
  template <typename... Args>
  ::c10::intrusive_ptr<::c10d::Work> CallWork(
    const char* operation, Args&&... args) {
    py::gil_scoped_acquire gil;
    return transport_->attr(operation)(std::forward<Args>(args)...)
      .template cast<::c10::intrusive_ptr<::c10d::Work>>();
  }

  std::optional<py::object> transport_;
};

::c10::intrusive_ptr<::c10d::Backend> CreateAxonBackend(
  const ::c10::intrusive_ptr<::c10d::Store>& store, int rank, int world_size,
  std::chrono::milliseconds timeout, py::object transport) {
  auto options = ::c10::make_intrusive<c10d::AxonBackend::Options>(
    std::make_shared<PythonAxonTransport>(std::move(transport)), timeout);
  return ::c10::make_intrusive<c10d::AxonBackend>(
    store, rank, world_size, options);
}

#ifdef AXON_C10D_UCX_ENABLED
::c10::intrusive_ptr<::c10d::Backend> CreateUcxBackend(
  const ::c10::intrusive_ptr<::c10d::Store>& store, int rank, int world_size,
  std::chrono::milliseconds timeout) {
  auto transport =
    std::make_shared<c10d::AxonUcxTransport>(store, rank, world_size, timeout);
  auto options = ::c10::make_intrusive<c10d::AxonBackend::Options>(
    std::move(transport), timeout);
  return ::c10::make_intrusive<c10d::AxonBackend>(
    store, rank, world_size, options);
}
#endif

}  // namespace eux::axon::python

PYBIND11_MODULE(_axon_c10d, module) {
  py::module_::import("torch._C._distributed_c10d");
  module.doc() = "PyTorch c10d bindings for the Axon backend";
  module.def(
    "create_backend", &eux::axon::python::CreateAxonBackend, py::arg("store"),
    py::arg("rank"), py::arg("world_size"), py::arg("timeout"),
    py::arg("transport"));
#ifdef AXON_C10D_UCX_ENABLED
  module.def(
    "create_ucx_backend", &eux::axon::python::CreateUcxBackend,
    py::arg("store"), py::arg("rank"), py::arg("world_size"),
    py::arg("timeout"));
#endif
}
