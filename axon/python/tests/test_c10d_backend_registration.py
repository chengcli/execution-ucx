"""End-to-end test for registering Axon with torch.distributed."""

import importlib
import pathlib
import sys
import types

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import FakeProcessGroup


def load_registration_helper(source_dir: pathlib.Path, extension_dir: pathlib.Path):
    sys.path.insert(0, str(extension_dir))
    native = importlib.import_module("_axon_c10d")

    package = types.ModuleType("axon")
    package.__path__ = [str(source_dir / "axon")]  # type: ignore[attr-defined]
    sys.modules["axon"] = package
    sys.modules["axon._axon_c10d"] = native
    return importlib.import_module("axon.distributed")


class FakeTransport:
    """Use PyTorch's in-process backend to validate the Axon c10d bridge."""

    def __init__(self, store, rank, world_size, timeout):
        del store, timeout
        self.backend = FakeProcessGroup._create_internal(rank, world_size)

    def send(self, tensors, dst_rank, tag):
        return self.backend.send(tensors, dst_rank, tag)

    def recv(self, tensors, src_rank, tag):
        return self.backend.recv(tensors, src_rank, tag)

    def barrier(self, options):
        return self.backend.barrier(options)

    def allreduce(self, tensors, options):
        return self.backend.allreduce(tensors, options)

    def reduce(self, tensors, options):
        return self.backend.reduce(tensors, options)


def main():
    source_dir = pathlib.Path(sys.argv[1])
    extension_dir = pathlib.Path(sys.argv[2])
    helper = load_registration_helper(source_dir, extension_dir)
    helper.register_backend(FakeTransport, name="axon_test", devices="cpu")

    store = dist.HashStore()
    dist.init_process_group("axon_test", store=store, rank=0, world_size=1)
    try:
        value = torch.tensor([7.0])
        dist.all_reduce(value)
        assert value.item() == 7.0
        dist.reduce(value, dst=0)
        dist.barrier()

        backend = dist.distributed_c10d._get_default_group()._get_backend(
            torch.device("cpu")
        )
        backend.send([value], 0, 11).wait()
        backend.recv([value], 0, 12).wait()

        assert dist.get_backend() == "axon_test"
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
