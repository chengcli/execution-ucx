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

#include "axon/c10d/axon_ucx_transport.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <c10/util/Exception.h>

#ifdef AXON_C10D_CUDA_ENABLED
#include <ATen/cuda/CUDAContext.h>
#endif

namespace eux::axon::c10d {
namespace {

class CompletedWork final : public ::c10d::Work {
 public:
  CompletedWork(int rank, ::c10d::OpType type, std::exception_ptr error)
    : ::c10d::Work(rank, type) {
    finish(std::move(error));
  }
};

void CheckStatus(ucs_status_t status, const char* operation) {
  TORCH_CHECK(
    status == UCS_OK, operation, " failed: ", ucs_status_string(status));
}

std::string AddressKey(int rank) {
  return "axon-ucx/address/" + std::to_string(rank);
}

}  // namespace

AxonUcxTransport::AxonUcxTransport(
  ::c10::intrusive_ptr<::c10d::Store> store, int rank, int world_size,
  std::chrono::milliseconds timeout)
  : store_(std::move(store)),
    rank_(rank),
    world_size_(world_size),
    timeout_(timeout),
    endpoints_(world_size, nullptr) {
  TORCH_CHECK(store_.defined(), "AxonUcxTransport requires a c10d Store");

  ucp_params_t context_params{};
  context_params.field_mask = UCP_PARAM_FIELD_FEATURES;
  context_params.features = UCP_FEATURE_TAG;

  ucp_config_t* config = nullptr;
  CheckStatus(ucp_config_read(nullptr, nullptr, &config), "ucp_config_read");
  const auto init_status = ucp_init(&context_params, config, &context_);
  ucp_config_release(config);
  CheckStatus(init_status, "ucp_init");

  ucp_worker_params_t worker_params{};
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
  CheckStatus(
    ucp_worker_create(context_, &worker_params, &worker_),
    "ucp_worker_create");

  ucp_address_t* local_address = nullptr;
  size_t local_address_size = 0;
  CheckStatus(
    ucp_worker_get_address(worker_, &local_address, &local_address_size),
    "ucp_worker_get_address");
  store_->set(
    AddressKey(rank_),
    std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_address),
      reinterpret_cast<uint8_t*>(local_address) + local_address_size));
  ucp_worker_release_address(worker_, local_address);

  std::vector<std::string> keys;
  keys.reserve(world_size_);
  for (int peer = 0; peer < world_size_; ++peer) {
    keys.push_back(AddressKey(peer));
  }
  store_->wait(keys, timeout_);

  for (int peer = 0; peer < world_size_; ++peer) {
    auto address = store_->get(AddressKey(peer));
    ucp_ep_params_t endpoint_params{};
    endpoint_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    endpoint_params.address =
      reinterpret_cast<const ucp_address_t*>(address.data());
    CheckStatus(
      ucp_ep_create(worker_, &endpoint_params, &endpoints_[peer]),
      "ucp_ep_create");
  }
}

AxonUcxTransport::~AxonUcxTransport() {
  std::lock_guard lock(mutex_);
  for (auto endpoint : endpoints_) {
    if (!endpoint) {
      continue;
    }
    ucp_request_param_t params{};
    params.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    params.flags = UCP_EP_CLOSE_FLAG_FORCE;
    try {
      WaitRequest(ucp_ep_close_nbx(endpoint, &params));
    } catch (...) {
      // Destructors cannot report UCX endpoint close errors.
    }
  }
  if (worker_) {
    ucp_worker_destroy(worker_);
  }
  if (context_) {
    ucp_cleanup(context_);
  }
}

::c10::intrusive_ptr<::c10d::Work> AxonUcxTransport::Send(
  std::vector<at::Tensor>& tensors, int dst_rank, int tag) {
  return Run(::c10d::OpType::SEND, [&] {
    SendTensors(tensors, dst_rank, PointToPointTag(rank_, dst_rank, tag));
  });
}

::c10::intrusive_ptr<::c10d::Work> AxonUcxTransport::Recv(
  std::vector<at::Tensor>& tensors, int src_rank, int tag) {
  return Run(::c10d::OpType::RECV, [&] {
    RecvTensors(tensors, src_rank, PointToPointTag(src_rank, rank_, tag));
  });
}

::c10::intrusive_ptr<::c10d::Work> AxonUcxTransport::Barrier(
  const ::c10d::BarrierOptions&) {
  return Run(::c10d::OpType::BARRIER, [&] {
    const auto sequence = collective_sequence_.fetch_add(1);
    at::Tensor token = at::zeros({1}, at::TensorOptions().dtype(at::kByte));
    std::vector<at::Tensor> tensors{token};
    if (rank_ == 0) {
      for (int peer = 1; peer < world_size_; ++peer) {
        RecvTensors(tensors, peer, CollectiveTag(kBarrierKind, sequence, peer));
      }
      for (int peer = 1; peer < world_size_; ++peer) {
        SendTensors(tensors, peer, CollectiveTag(kBarrierKind, sequence, 0));
      }
    } else {
      SendTensors(tensors, 0, CollectiveTag(kBarrierKind, sequence, rank_));
      RecvTensors(tensors, 0, CollectiveTag(kBarrierKind, sequence, 0));
    }
  });
}

::c10::intrusive_ptr<::c10d::Work> AxonUcxTransport::Allreduce(
  std::vector<at::Tensor>& tensors, const ::c10d::AllreduceOptions& options) {
  return Run(::c10d::OpType::ALLREDUCE, [&] {
    ValidateSum(options.reduceOp);
    const auto sequence = collective_sequence_.fetch_add(1);
    if (rank_ == 0) {
      for (int peer = 1; peer < world_size_; ++peer) {
        std::vector<at::Tensor> incoming;
        incoming.reserve(tensors.size());
        for (const auto& tensor : tensors) {
          incoming.push_back(at::empty_like(tensor));
        }
        RecvTensors(
          incoming, peer, CollectiveTag(kReduceKind, sequence, peer));
        for (size_t i = 0; i < tensors.size(); ++i) {
          tensors[i].add_(incoming[i]);
        }
      }
      for (int peer = 1; peer < world_size_; ++peer) {
        SendTensors(
          tensors, peer, CollectiveTag(kBroadcastKind, sequence, 0));
      }
    } else {
      SendTensors(tensors, 0, CollectiveTag(kReduceKind, sequence, rank_));
      RecvTensors(tensors, 0, CollectiveTag(kBroadcastKind, sequence, 0));
    }
  });
}

::c10::intrusive_ptr<::c10d::Work> AxonUcxTransport::Reduce(
  std::vector<at::Tensor>& tensors, const ::c10d::ReduceOptions& options) {
  return Run(::c10d::OpType::REDUCE, [&] {
    ValidateSum(options.reduceOp);
    const auto sequence = collective_sequence_.fetch_add(1);
    const int root = static_cast<int>(options.rootRank);
    if (rank_ == root) {
      for (int peer = 0; peer < world_size_; ++peer) {
        if (peer == root) {
          continue;
        }
        std::vector<at::Tensor> incoming;
        incoming.reserve(tensors.size());
        for (const auto& tensor : tensors) {
          incoming.push_back(at::empty_like(tensor));
        }
        RecvTensors(
          incoming, peer, CollectiveTag(kReduceKind, sequence, peer));
        for (size_t i = 0; i < tensors.size(); ++i) {
          tensors[i].add_(incoming[i]);
        }
      }
    } else {
      SendTensors(tensors, root, CollectiveTag(kReduceKind, sequence, rank_));
    }
  });
}

::c10::intrusive_ptr<::c10d::Work> AxonUcxTransport::Run(
  ::c10d::OpType type, const std::function<void()>& operation) {
  try {
    std::lock_guard lock(mutex_);
    operation();
    return ::c10::make_intrusive<CompletedWork>(rank_, type, nullptr);
  } catch (...) {
    return ::c10::make_intrusive<CompletedWork>(
      rank_, type, std::current_exception());
  }
}

void AxonUcxTransport::SendTensors(
  const std::vector<at::Tensor>& tensors, int dst_rank, uint64_t tag) {
  ValidateTensors(tensors);
  SynchronizeTensors(tensors);
  for (size_t i = 0; i < tensors.size(); ++i) {
    ucp_request_param_t params{};
    WaitRequest(ucp_tag_send_nbx(
      endpoints_.at(dst_rank), tensors[i].data_ptr(), tensors[i].nbytes(),
      tag + i, &params));
  }
}

void AxonUcxTransport::RecvTensors(
  std::vector<at::Tensor>& tensors, int, uint64_t tag) {
  ValidateTensors(tensors);
  SynchronizeTensors(tensors);
  for (size_t i = 0; i < tensors.size(); ++i) {
    ucp_request_param_t params{};
    WaitRequest(ucp_tag_recv_nbx(
      worker_, tensors[i].data_ptr(), tensors[i].nbytes(), tag + i,
      UINT64_MAX, &params));
  }
}

void AxonUcxTransport::SynchronizeTensors(
  const std::vector<at::Tensor>& tensors) const {
#ifdef AXON_C10D_CUDA_ENABLED
  for (const auto& tensor : tensors) {
    if (tensor.is_cuda()) {
      at::cuda::getCurrentCUDAStream(tensor.device().index()).synchronize();
    }
  }
#else
  (void)tensors;
#endif
}

void AxonUcxTransport::WaitRequest(ucs_status_ptr_t request) {
  if (request == nullptr) {
    return;
  }
  if (UCS_PTR_IS_ERR(request)) {
    CheckStatus(UCS_PTR_STATUS(request), "UCX operation");
  }
  ucs_status_t status;
  do {
    ucp_worker_progress(worker_);
    status = ucp_request_check_status(request);
  } while (status == UCS_INPROGRESS);
  ucp_request_free(request);
  CheckStatus(status, "UCX operation");
}

void AxonUcxTransport::ValidateTensors(
  const std::vector<at::Tensor>& tensors) const {
  TORCH_CHECK(!tensors.empty(), "AxonUcxTransport requires tensors");
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
      tensor.device().is_cpu() || tensor.is_cuda(),
      "AxonUcxTransport supports CPU and CUDA tensors only");
#ifndef AXON_C10D_CUDA_ENABLED
    TORCH_CHECK(
      !tensor.is_cuda(),
      "AxonUcxTransport CUDA support requires EXECUTION_UCX_ENABLE_CUDA=ON");
#endif
    TORCH_CHECK(
      tensor.is_contiguous(), "AxonUcxTransport requires contiguous tensors");
  }
}

void AxonUcxTransport::ValidateSum(const ::c10d::ReduceOp& op) const {
  TORCH_CHECK(
    op.op_ == ::c10d::ReduceOp::SUM,
    "AxonUcxTransport currently supports SUM reductions only");
}

uint64_t AxonUcxTransport::PointToPointTag(
  int src_rank, int dst_rank, int tag) const {
  return (kPointToPointKind << 60) | (static_cast<uint64_t>(src_rank) << 48)
         | (static_cast<uint64_t>(dst_rank) << 36)
         | (static_cast<uint32_t>(tag) & 0xffffffffULL);
}

uint64_t AxonUcxTransport::CollectiveTag(
  uint64_t kind, uint64_t sequence, int rank) const {
  return (kind << 60) | ((sequence & 0x0fffffffULL) << 28)
         | static_cast<uint32_t>(rank);
}

}  // namespace eux::axon::c10d
