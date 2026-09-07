#!/usr/bin/env python3
"""
Create ZIP checkpoints that exercise container details torch.save leaves to the caller.

torch is not required: a stub `torch` module provides the two names the pickle refers to
(`torch.FloatStorage` and `torch._utils._rebuild_tensor_v2`), and the archive is written the
way `torch.save` writes it (every entry under a directory named after the file).

  protocol4.pt   pickled with protocol 4 (FRAME, SHORT_BINUNICODE, MEMOIZE, STACK_GLOBAL)
  big_endian.pt  a `byteorder` entry of "big", which the reader must refuse

Run with: python3 create_protocol4.py
"""

import io
import pickle
import struct
import sys
import types
import zipfile
from collections import OrderedDict


def install_torch_stub():
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        raise NotImplementedError

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch.FloatStorage = FloatStorage
    torch._utils = utils
    sys.modules["torch"] = torch
    sys.modules["torch._utils"] = utils
    return torch


torch = install_torch_stub()


class Storage:
    def __init__(self, key, values):
        self.key = key
        self.values = values


class Tensor:
    def __init__(self, storage, shape, stride, offset=0):
        self.storage = storage
        self.shape = shape
        self.stride = stride
        self.offset = offset

    def __reduce_ex__(self, protocol):
        return (
            torch._utils._rebuild_tensor_v2,
            (self.storage, self.offset, self.shape, self.stride, False, OrderedDict()),
        )


class Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, Storage):
            return ("storage", torch.FloatStorage, obj.key, "cpu", len(obj.values))
        return None


def write_checkpoint(filename, state_dict, storages, protocol, byteorder="little"):
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    buffer = io.BytesIO()
    Pickler(buffer, protocol=protocol).dump(state_dict)

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{stem}/data.pkl", buffer.getvalue())
        archive.writestr(f"{stem}/byteorder", byteorder)
        for storage in storages:
            data = b"".join(struct.pack("<f", v) for v in storage.values)
            archive.writestr(f"{stem}/data/{storage.key}", data)
        archive.writestr(f"{stem}/version", "3\n")
    print(f"Created {filename}")


def main():
    weight_storage = Storage("0", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bias_storage = Storage("1", [0.5, -0.5])
    state_dict = OrderedDict(
        [
            ("weight", Tensor(weight_storage, (2, 3), (3, 1))),
            ("bias", Tensor(bias_storage, (2,), (1,))),
        ]
    )
    write_checkpoint("test_data/protocol4.pt", state_dict, [weight_storage, bias_storage], protocol=4)
    write_checkpoint(
        "test_data/big_endian.pt",
        OrderedDict([("bias", Tensor(bias_storage, (2,), (1,)))]),
        [bias_storage],
        protocol=2,
        byteorder="big",
    )


if __name__ == "__main__":
    main()
