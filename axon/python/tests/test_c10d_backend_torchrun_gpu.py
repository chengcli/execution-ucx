"""Two-process torchrun test for direct Axon UCX CUDA tensor transport."""

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
    assert torch.cuda.device_count() >= 2
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    source_dir = pathlib.Path(sys.argv[1])
    extension_dir = pathlib.Path(sys.argv[2])
    helper = load_registration_helper(source_dir, extension_dir)
    helper.register_ucx_backend(name="axon_ucx_gpu", devices="cuda")

    dist.init_process_group("axon_ucx_gpu", init_method="env://")
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 2
        stream = torch.cuda.Stream(device=device)

        with torch.cuda.stream(stream):
            value = torch.full((1024 * 1024,), float(rank + 1), device=device)
            dist.all_reduce(value)
        stream.synchronize()
        torch.testing.assert_close(value, torch.full_like(value, 3.0))

        with torch.cuda.stream(stream):
            reduced = torch.full((1024,), float(rank + 1), device=device)
            dist.reduce(reduced, dst=0)
        stream.synchronize()
        if rank == 0:
            torch.testing.assert_close(reduced, torch.full_like(reduced, 3.0))

        with torch.cuda.stream(stream):
            message = torch.full(
                (1024 * 1024,), -1, dtype=torch.int64, device=device
            )
            if rank == 0:
                message.fill_(1234)
                dist.send(message, dst=1, tag=91)
            else:
                dist.recv(message, src=0, tag=91)
        stream.synchronize()
        if rank == 1:
            torch.testing.assert_close(message, torch.full_like(message, 1234))

        dist.barrier()
        assert dist.get_backend() == "axon_ucx_gpu"
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
