#!/usr/bin/env python3
"""
Create TAR format test fixtures for the burn-store PyTorch reader.

The TAR container was written by PyTorch before 0.1.10. Modern torch.save cannot produce
it, so the files are built by hand following the reader in torch/serialization.py
(`_legacy_load`, the `tarfile` branch), which is the only surviving description of the
layout:

  sys_info  pickle: {protocol_version, little_endian, type_sizes}
  storages  pickle: count
            per storage: pickle (key, location, storage class), i64 numel, raw bytes
            pickle: list of (view key, root key, offset, numel) storage views
  tensors   pickle: count
            per tensor: pickle (key, storage key, tensor class),
                        i32 ndim, 4 unused bytes, ndim x i64 size, ndim x i64 stride,
                        i64 storage offset
  pickle    the saved object; every tensor is a persistent id naming a `tensors` entry

Run with: python3 create_tar_format.py
"""

import io
import os
import pickle
import struct
import tarfile
from collections import OrderedDict


STORAGE_FORMATS = {
    "FloatStorage": ("<f", "FloatTensor"),
    "DoubleStorage": ("<d", "DoubleTensor"),
    "LongStorage": ("<q", "LongTensor"),
    "IntStorage": ("<i", "IntTensor"),
    "ShortStorage": ("<h", "ShortTensor"),
    "ByteStorage": ("<B", "ByteTensor"),
    "CharStorage": ("<b", "CharTensor"),
}


def pickle_int(value):
    if 0 <= value < 256:
        return b"K" + bytes([value])
    if 0 <= value < 65536:
        return b"M" + struct.pack("<H", value)
    return b"J" + struct.pack("<i", value)


def pickle_tuple_with_torch_class(items, class_name):
    """
    Pickle `(*items, torch.<class_name>)` by hand, as the GLOBAL reference to a torch
    class cannot be produced without torch installed.
    """
    out = io.BytesIO()
    out.write(b"\x80\x02(")
    for item in items:
        if isinstance(item, int):
            out.write(pickle_int(item))
        else:
            encoded = item.encode("utf-8")
            out.write(b"U" + bytes([len(encoded)]) + encoded)
    out.write(b"ctorch\n" + class_name.encode("ascii") + b"\nt.")
    return out.getvalue()


def row_major_stride(shape):
    stride = []
    step = 1
    for dim in reversed(shape):
        stride.insert(0, step)
        step *= dim
    return stride


def create_tar_pytorch_file(filename, tensors, dtypes, views=()):
    """
    Args:
        filename: Output path
        tensors: name -> (values, shape)
        dtypes: name -> storage class name
        views: (name, root name, element offset, element count, shape) tuples; each adds a
            tensor over a storage view of the root tensor's storage
    """
    storages = io.BytesIO()
    tensor_table = io.BytesIO()
    pickle.dump(len(tensors), storages, protocol=2)
    pickle.dump(len(tensors) + len(views), tensor_table, protocol=2)

    state_dict = OrderedDict()
    for index, (name, (values, shape)) in enumerate(tensors.items()):
        storage_key = 1000 + index
        tensor_key = 2000 + index
        storage_type = dtypes[name]
        fmt, tensor_type = STORAGE_FORMATS[storage_type]

        storages.write(pickle_tuple_with_torch_class([storage_key, "cpu"], storage_type))
        storages.write(struct.pack("<q", len(values)))
        storages.write(b"".join(struct.pack(fmt, v) for v in values))

        tensor_table.write(pickle_tuple_with_torch_class([tensor_key, storage_key], tensor_type))
        tensor_table.write(struct.pack("<i", len(shape)))
        tensor_table.write(b"\x00" * 4)
        tensor_table.write(struct.pack(f"<{len(shape)}q", *shape))
        tensor_table.write(struct.pack(f"<{len(shape)}q", *row_major_stride(shape)))
        tensor_table.write(struct.pack("<q", 0))

        state_dict[name] = _PersistentTensor(tensor_key)

    names = list(tensors)
    view_entries = []
    for index, (name, root_name, offset, numel, shape) in enumerate(views):
        root_index = names.index(root_name)
        view_key = 3000 + index
        tensor_key = 4000 + index
        _, tensor_type = STORAGE_FORMATS[dtypes[root_name]]
        view_entries.append((view_key, 1000 + root_index, offset, numel))
        tensor_table.write(pickle_tuple_with_torch_class([tensor_key, view_key], tensor_type))
        tensor_table.write(struct.pack("<i", len(shape)))
        tensor_table.write(b"\x00" * 4)
        tensor_table.write(struct.pack(f"<{len(shape)}q", *shape))
        tensor_table.write(struct.pack(f"<{len(shape)}q", *row_major_stride(shape)))
        tensor_table.write(struct.pack("<q", 0))
        state_dict[name] = _PersistentTensor(tensor_key)

    pickle.dump(view_entries, storages, protocol=2)

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, _PersistentTensor):
                return str(obj.key)
            return None

    main = io.BytesIO()
    Pickler(main, protocol=2).dump(state_dict)

    sys_info = pickle.dumps(
        {
            "protocol_version": 1000,
            "little_endian": True,
            "type_sizes": {"short": 2, "int": 4, "long": 8},
        },
        protocol=2,
    )

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with tarfile.open(filename, "w") as tar:
        for entry_name, data in [
            ("sys_info", sys_info),
            ("pickle", main.getvalue()),
            ("tensors", tensor_table.getvalue()),
            ("storages", storages.getvalue()),
        ]:
            info = tarfile.TarInfo(name=entry_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    print(f"Created {filename} ({os.path.getsize(filename)} bytes): {list(tensors)}")


class _PersistentTensor:
    def __init__(self, key):
        self.key = key


def main():
    os.makedirs("test_data", exist_ok=True)

    create_tar_pytorch_file(
        "test_data/tar_float32.tar",
        {"tensor": ([1.0, 2.5, -3.7, 0.0], [4])},
        {"tensor": "FloatStorage"},
    )
    create_tar_pytorch_file(
        "test_data/tar_float64.tar",
        {"tensor": ([1.1, 2.2, 3.3], [3])},
        {"tensor": "DoubleStorage"},
    )
    create_tar_pytorch_file(
        "test_data/tar_int64.tar",
        {"tensor": ([100, -200, 300, 0], [4])},
        {"tensor": "LongStorage"},
    )
    create_tar_pytorch_file(
        "test_data/tar_weight_bias.tar",
        {
            "weight": ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]),
            "bias": ([0.01, 0.02], [2]),
        },
        {"weight": "FloatStorage", "bias": "FloatStorage"},
    )
    create_tar_pytorch_file(
        "test_data/tar_multi_dtype.tar",
        {
            "float_tensor": ([1.5, 2.5, 3.5], [3]),
            "double_tensor": ([1.111, 2.222], [2]),
            "int_tensor": ([10, 20, 30, 40], [4]),
        },
        {
            "float_tensor": "FloatStorage",
            "double_tensor": "DoubleStorage",
            "int_tensor": "LongStorage",
        },
    )
    create_tar_pytorch_file(
        "test_data/tar_storage_view.tar",
        {"root": ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8])},
        {"root": "FloatStorage"},
        views=[("window", "root", 2, 4, [2, 2])],
    )
    create_tar_pytorch_file(
        "test_data/tar_2d_tensor.tar",
        {"matrix": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 4])},
        {"matrix": "FloatStorage"},
    )


if __name__ == "__main__":
    main()
