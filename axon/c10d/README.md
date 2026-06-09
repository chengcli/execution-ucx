# Axon c10d Backend

`execution-ucx::axon_c10d` exposes Axon transports through PyTorch's
`c10d::Backend` interface. It is intended for applications such as Snapy that
construct a generic `c10d::ProcessGroup` and install a backend per device type.

The adapter implements the operation surface currently used by Snapy:

- `send`
- `recv`
- `barrier`
- `allreduce`
- `reduce`

An `AxonTransport` implementation owns rendezvous, Axon/UCX progress, tensor
memory handling, and completion of the returned `c10d::Work`. This separation
keeps PyTorch types out of Axon's core runtime and keeps ProcessGroup policy out
of the transport.

## Build

```sh
cmake -S . -B build \
  -DEXECUTION_UCX_BUILD_AXON_C10D=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build --target execution_ucx_axon_c10d
```

## ProcessGroup Integration

The installation helper follows the same pattern as Snapy's Gloo and NCCL
initializers:

```cpp
#include <axon/c10d/axon_backend.hpp>

auto pg = c10::make_intrusive<c10d::ProcessGroup>(store, rank, world_size);
auto transport = std::make_shared<AxonTx>(store, rank, world_size);

eux::axon::c10d::InstallAxonBackend(
  pg, store, transport,
  {c10::DeviceType::CPU, c10::DeviceType::CUDA});
```

The same backend instance is registered for each listed device type. A
transport should only advertise device types whose memory it can access.

## Python Registration

Build the PyTorch binding with:

```sh
cmake -S . -B build \
  -DEXECUTION_UCX_BUILD_AXON_C10D=ON \
  -DEXECUTION_UCX_BUILD_AXON_C10D_PYTHON=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

Then register a transport factory before initializing `torch.distributed`:

```python
import axon
import torch.distributed as dist

axon.register_torch_backend(AxonTx, devices=("cpu", "cuda"))
dist.init_process_group("axon")
```

The factory receives `(store, rank, world_size, timeout)`. Its returned
transport object must implement `send`, `recv`, `barrier`, `allreduce`, and
`reduce`, with each method returning a PyTorch distributed `Work` object.

The registration works with `torchrun` when every worker registers the backend
before calling `init_process_group`:

```python
axon.register_torch_ucx_backend()
dist.init_process_group("axon", init_method="env://")
```

```sh
torchrun --standalone --nproc-per-node=2 application.py
```

`torchrun` supplies the rendezvous store, rank, world size, and timeout to each
transport factory. Build the concrete UCX transport and bindings with
`EXECUTION_UCX_BUILD_AXON_C10D_UCX=ON` and
`EXECUTION_UCX_BUILD_AXON_C10D_PYTHON=ON`. Enable direct CUDA tensor transport
with `EXECUTION_UCX_ENABLE_CUDA=ON`. The transport currently supports contiguous
CPU and CUDA tensors and SUM reductions.

## C++ ProcessGroup examples

Build the standalone CPU and GPU examples with:

```sh
cmake -S . -B build \
  -DEXECUTION_UCX_BUILD_AXON_C10D=ON \
  -DEXECUTION_UCX_BUILD_AXON_C10D_UCX=ON \
  -DEXECUTION_UCX_BUILD_AXON_C10D_EXAMPLES=ON \
  -DEXECUTION_UCX_ENABLE_CUDA=ON
cmake --build build
```

Launch rank 0 and rank 1 in separate processes using the same TCPStore address:

```sh
build/execution_ucx_axon_c10d_cpu_example 0 127.0.0.1 29500 &
build/execution_ucx_axon_c10d_cpu_example 1 127.0.0.1 29500

UCX_TLS=sm,cuda_copy,cuda_ipc \
  build/execution_ucx_axon_c10d_gpu_example 0 127.0.0.1 29501 &
UCX_TLS=sm,cuda_copy,cuda_ipc \
  build/execution_ucx_axon_c10d_gpu_example 1 127.0.0.1 29501
```
