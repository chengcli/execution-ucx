"""PyTorch distributed registration helpers for the Axon c10d backend."""

from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any


TransportFactory = Callable[[Any, int, int, timedelta], Any]


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

    try:
        from . import _axon_c10d
    except ImportError as error:
        raise ImportError(
            "Axon c10d bindings are unavailable. Build with "
            "EXECUTION_UCX_BUILD_AXON_C10D_PYTHON=ON."
        ) from error

    device_list = [devices] if isinstance(devices, str) else list(devices)
    if not device_list:
        raise ValueError("devices must contain at least one device type")

    def create_backend(store, rank, world_size, timeout):
        transport = transport_factory(store, rank, world_size, timeout)
        return _axon_c10d.create_backend(
            store, rank, world_size, timeout, transport
        )

    dist.Backend.register_backend(name, create_backend, devices=device_list)
