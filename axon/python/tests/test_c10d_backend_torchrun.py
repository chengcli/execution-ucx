"""Two-process torchrun test for the concrete Axon UCX transport."""

import importlib
import os
import pathlib
import sys
import types

import torch
import torch.distributed as dist


def load_registration_helper(source_dir: pathlib.Path, extension_dir: pathlib.Path):
    sys.path.insert(0, str(extension_dir))
    native = importlib.import_module("_axon_c10d")

    package = types.ModuleType("axon")
    package.__path__ = [str(source_dir / "axon")]  # type: ignore[attr-defined]
    sys.modules["axon"] = package
    sys.modules["axon._axon_c10d"] = native
    return importlib.import_module("axon.distributed")


def main():
    source_dir = pathlib.Path(sys.argv[1])
    extension_dir = pathlib.Path(sys.argv[2])
    helper = load_registration_helper(source_dir, extension_dir)
    helper.register_ucx_backend(name="axon_torchrun", devices="cpu")

    dist.init_process_group("axon_torchrun", init_method="env://")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert rank == int(os.environ["RANK"])
        assert world_size == int(os.environ["WORLD_SIZE"]) == 2

        value = torch.tensor([float(rank + 1)])
        dist.all_reduce(value)
        assert value.item() == 3.0

        reduced = torch.tensor([float(rank + 1)])
        dist.reduce(reduced, dst=0)
        if rank == 0:
            assert reduced.item() == 3.0

        message = torch.tensor([-1], dtype=torch.int64)
        if rank == 0:
            message.fill_(1234)
            dist.send(message, dst=1, tag=77)
        else:
            dist.recv(message, src=0, tag=77)
            assert message.item() == 1234

        dist.barrier()
        assert dist.get_backend() == "axon_torchrun"
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
