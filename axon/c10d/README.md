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
auto transport = std::make_shared<MyAxonTransport>(store, rank, world_size);

eux::axon::c10d::InstallAxonBackend(
  pg, store, transport,
  {c10::DeviceType::CPU, c10::DeviceType::CUDA});
```

The same backend instance is registered for each listed device type. A
transport should only advertise device types whose memory it can access.
