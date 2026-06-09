"""PyTorch distributed registration helpers for the Axon c10d backend."""

from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any


TransportFactory = Callable[[Any, int, int, timedelta], Any]


def _load_bindings():
    try:
        from . import _axon_c10d
    except ImportError as error:
        raise ImportError(
            "Axon c10d bindings are unavailable. Build with "
            "EXECUTION_UCX_BUILD_AXON_C10D_PYTHON=ON."
        ) from error
    return _axon_c10d


def _device_list(devices: str | Iterable[str]) -> list[str]:
    device_list = [devices] if isinstance(devices, str) else list(devices)
    if not device_list:
        raise ValueError("devices must contain at least one device type")
    return device_list


def register_backend(
    transport_factory: TransportFactory,
    *,
    name: str = "axon",
    devices: str | Iterable[str] = ("cpu", "cuda"),
) -> None:
    """Register Axon as a PyTorch distributed backend.

    ``transport_factory`` receives ``(store, rank, world_size, timeout)`` and
    must return an object implementing ``send``, ``recv``, ``barrier``,
    ``allreduce``, and ``reduce``. Each operation must return a PyTorch
    distributed ``Work`` object.
    """

    import torch.distributed as dist

    _axon_c10d = _load_bindings()
    device_list = _device_list(devices)

    def create_backend(store, rank, world_size, timeout):
        transport = transport_factory(store, rank, world_size, timeout)
        return _axon_c10d.create_backend(
            store, rank, world_size, timeout, transport
        )

    dist.Backend.register_backend(name, create_backend, devices=device_list)


def register_ucx_backend(
    *, name: str = "axon", devices: str | Iterable[str] = ("cpu", "cuda")
) -> None:
    """Register the built-in UCX transport for contiguous CPU/CUDA tensors."""

    import torch.distributed as dist

    _axon_c10d = _load_bindings()
    if not hasattr(_axon_c10d, "create_ucx_backend"):
        raise ImportError(
            "The Axon UCX transport is unavailable. Build with "
            "EXECUTION_UCX_BUILD_AXON_C10D_UCX=ON."
        )

    dist.Backend.register_backend(
        name, _axon_c10d.create_ucx_backend, devices=_device_list(devices)
    )
