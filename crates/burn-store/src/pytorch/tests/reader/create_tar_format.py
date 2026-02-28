#!/usr/bin/env python3
"""
Create TAR format test fixtures for burn-store integration tests.

The TAR format was used by very early versions of PyTorch (pre 0.1.10).
Modern torch.save cannot create this format, so we construct it manually.

TAR format structure:
  - sys_info: pickle with {protocol_version, little_endian, type_sizes}
  - pickle: pickle with OrderedDict containing _rebuild_tensor_v2 REDUCE calls
  - storages: count_pickle + for each storage: (key, device, class) pickle + u64 num_elements + raw data
"""

import io
import pickle
import struct
import tarfile
import os
from collections import OrderedDict


def create_sys_info():
    """Create sys_info pickle data."""
    sys_info = {
        "protocol_version": 1000,
        "little_endian": True,
        "type_sizes": {
            "short": 2,
            "int": 4,
            "long": 8,
        },
    }
    return pickle.dumps(sys_info, protocol=2)


def encode_tensor_data(values: list, storage_type: str) -> tuple:
    """Encode tensor values to bytes and return (bytes, element_size)."""
    fmt_map = {
        "FloatStorage": ("<f", 4),
        "DoubleStorage": ("<d", 8),
        "LongStorage": ("<q", 8),
        "IntStorage": ("<i", 4),
        "ShortStorage": ("<h", 2),
        "ByteStorage": ("<B", 1),
        "CharStorage": ("<b", 1),
        "BoolStorage": ("<B", 1),
        "HalfStorage": ("<e", 2),
    }
    fmt, size = fmt_map[storage_type]
    data = b"".join(struct.pack(fmt, v) for v in values)
    return data, size


def write_int(buffer, value):
    """Write an integer using appropriate pickle opcode."""
    if 0 <= value < 256:
        buffer.write(b'K')  # BININT1
        buffer.write(bytes([value]))
    elif 0 <= value < 65536:
        buffer.write(b'M')  # BININT2
        buffer.write(struct.pack('<H', value))
    else:
        buffer.write(b'J')  # BININT
        buffer.write(struct.pack('<i', value))


def write_string(buffer, s):
    """Write a string using appropriate pickle opcode."""
    s_bytes = s.encode('utf-8')
    if len(s_bytes) < 256:
        buffer.write(b'U')  # SHORT_BINSTRING
        buffer.write(bytes([len(s_bytes)]))
        buffer.write(s_bytes)
    else:
        buffer.write(b'T')  # BINSTRING
        buffer.write(struct.pack('<I', len(s_bytes)))
        buffer.write(s_bytes)


def create_storages_blob_manual(tensors: list) -> bytes:
    """
    Create the storages binary blob manually.

    Args:
        tensors: List of (key, storage_type, element_size, data_bytes) tuples
    """
    buffer = io.BytesIO()

    # Write storage count as pickle (simple integer)
    pickle.dump(len(tensors), buffer, protocol=2)

    for key, storage_type, element_size, data_bytes in tensors:
        # Manually construct the tuple pickle with GLOBAL class reference
        # Format: (key, "cpu", <class 'torch.FloatStorage'>)

        tuple_buffer = io.BytesIO()
        # Protocol 2 header
        tuple_buffer.write(b'\x80\x02')

        # Build tuple with MARK + items + TUPLE
        tuple_buffer.write(b'(')  # MARK

        # First item: storage key (string)
        write_string(tuple_buffer, key)

        # Second item: device "cpu"
        tuple_buffer.write(b'U\x03cpu')

        # Third item: class reference using GLOBAL
        tuple_buffer.write(b'c')  # GLOBAL opcode
        tuple_buffer.write(b'torch\n')  # module
        tuple_buffer.write(storage_type.encode('ascii') + b'\n')  # name

        # End tuple
        tuple_buffer.write(b't')  # TUPLE
        tuple_buffer.write(b'.')  # STOP

        buffer.write(tuple_buffer.getvalue())

        # Write num_elements as u64 little-endian
        num_elements = len(data_bytes) // element_size
        buffer.write(struct.pack("<Q", num_elements))

        # Write raw data
        buffer.write(data_bytes)

    return buffer.getvalue()


def create_main_pickle_manual(tensors_info: list) -> bytes:
    """
    Create the main pickle containing _rebuild_tensor_v2 REDUCE calls.

    For each tensor, we need:
    - GLOBAL torch._utils _rebuild_tensor_v2
    - MARK
    - args tuple: (persistent_id, offset, shape, stride, requires_grad, hooks)
    - TUPLE
    - REDUCE

    The persistent_id is a PersistentTuple: ('storage', <class>, key, device, num_elements)
    """
    buffer = io.BytesIO()

    # Protocol 2 header
    buffer.write(b'\x80\x02')

    # Build OrderedDict: GLOBAL + EMPTY_LIST + items + TUPLE + REDUCE
    # OrderedDict([('name1', tensor1), ('name2', tensor2)])

    # GLOBAL collections OrderedDict
    buffer.write(b'ccollections\nOrderedDict\n')

    # Start list for items
    buffer.write(b'(')  # MARK
    buffer.write(b']')  # EMPTY_LIST

    # For each tensor, add (name, rebuilt_tensor) to the list
    for name, storage_key, storage_type, shape, num_elements in tensors_info:
        # Calculate stride for row-major (C) order
        stride = []
        s = 1
        for dim in reversed(shape):
            stride.insert(0, s)
            s *= dim

        # Build inner tuple: (name, tensor_value)
        buffer.write(b'(')  # MARK for (name, value) tuple

        # Write name
        write_string(buffer, name)

        # Now build the tensor using _rebuild_tensor_v2 REDUCE
        # GLOBAL torch._utils _rebuild_tensor_v2
        buffer.write(b'ctorch._utils\n_rebuild_tensor_v2\n')

        # Build args tuple for _rebuild_tensor_v2
        # (persistent_id, offset, shape, stride, requires_grad, backward_hooks)
        buffer.write(b'(')  # MARK for args tuple

        # arg 0: persistent_id tuple: ('storage', class, key, device, num_elements)
        # This will be converted to PersistentTuple by the reader
        buffer.write(b'(')  # MARK for persistent_id

        write_string(buffer, 'storage')

        # Class reference - GLOBAL torch FloatStorage
        buffer.write(b'c')
        buffer.write(b'torch\n')
        buffer.write(storage_type.encode('ascii') + b'\n')

        # Storage key
        write_string(buffer, storage_key)

        # Device
        buffer.write(b'U\x03cpu')

        # num_elements
        write_int(buffer, num_elements)

        buffer.write(b't')  # TUPLE - end persistent_id

        # arg 1: storage offset (0)
        buffer.write(b'K\x00')

        # arg 2: shape tuple
        buffer.write(b'(')
        for dim in shape:
            write_int(buffer, dim)
        buffer.write(b't')

        # arg 3: stride tuple
        buffer.write(b'(')
        for s_val in stride:
            write_int(buffer, s_val)
        buffer.write(b't')

        # arg 4: requires_grad (False)
        buffer.write(b'\x89')  # NEWFALSE

        # arg 5: backward_hooks (empty OrderedDict)
        buffer.write(b'ccollections\nOrderedDict\n')
        buffer.write(b'(')
        buffer.write(b']')
        buffer.write(b't')
        buffer.write(b'R')  # REDUCE to create empty OrderedDict

        buffer.write(b't')  # TUPLE - end args tuple

        buffer.write(b'R')  # REDUCE - call _rebuild_tensor_v2 with args

        buffer.write(b't')  # TUPLE - end (name, tensor) tuple

        buffer.write(b'a')  # APPEND to list

    buffer.write(b't')  # TUPLE - wrap list in tuple for REDUCE
    buffer.write(b'R')  # REDUCE - call OrderedDict with the list
    buffer.write(b'.')  # STOP

    return buffer.getvalue()


def create_tar_pytorch_file(filename: str, tensors: dict, dtypes: dict):
    """
    Create a TAR format PyTorch file.

    Args:
        filename: Output file path
        tensors: Dict of tensor_name -> (values_list, shape)
        dtypes: Dict of tensor_name -> storage_type
    """
    # Prepare storage data
    storage_list = []  # (key, storage_type, element_size, data_bytes)
    tensors_info = []  # (name, storage_key, storage_type, shape, num_elements)

    for idx, (name, (values, shape)) in enumerate(tensors.items()):
        storage_key = str(idx)
        storage_type = dtypes[name]
        data_bytes, element_size = encode_tensor_data(values, storage_type)
        num_elements = len(values)

        storage_list.append((storage_key, storage_type, element_size, data_bytes))
        tensors_info.append((name, storage_key, storage_type, shape, num_elements))

    # Create the three main entries
    sys_info_data = create_sys_info()
    pickle_data = create_main_pickle_manual(tensors_info)
    storages_data = create_storages_blob_manual(storage_list)

    # Write TAR archive
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with tarfile.open(filename, "w") as tar:
        # Add sys_info
        tarinfo = tarfile.TarInfo(name="sys_info")
        tarinfo.size = len(sys_info_data)
        tar.addfile(tarinfo, io.BytesIO(sys_info_data))

        # Add pickle
        tarinfo = tarfile.TarInfo(name="pickle")
        tarinfo.size = len(pickle_data)
        tar.addfile(tarinfo, io.BytesIO(pickle_data))

        # Add storages
        tarinfo = tarfile.TarInfo(name="storages")
        tarinfo.size = len(storages_data)
        tar.addfile(tarinfo, io.BytesIO(storages_data))

    size = os.path.getsize(filename)
    print(f"Created {filename} ({size} bytes)")
    print(f"  Tensors: {list(tensors.keys())}")


def main():
    # Create test_data directory
    os.makedirs("test_data", exist_ok=True)

    # Test 1: Single float32 tensor
    create_tar_pytorch_file(
        "test_data/tar_float32.tar",
        {"tensor": ([1.0, 2.5, -3.7, 0.0], [4])},
        {"tensor": "FloatStorage"},
    )

    # Test 2: Single float64 tensor
    create_tar_pytorch_file(
        "test_data/tar_float64.tar",
        {"tensor": ([1.1, 2.2, 3.3], [3])},
        {"tensor": "DoubleStorage"},
    )

    # Test 3: Single int64 tensor
    create_tar_pytorch_file(
        "test_data/tar_int64.tar",
        {"tensor": ([100, -200, 300, 0], [4])},
        {"tensor": "LongStorage"},
    )

    # Test 4: Multiple tensors (weight + bias)
    create_tar_pytorch_file(
        "test_data/tar_weight_bias.tar",
        {
            "weight": ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]),
            "bias": ([0.01, 0.02], [2]),
        },
        {
            "weight": "FloatStorage",
            "bias": "FloatStorage",
        },
    )

    # Test 5: Different dtypes in one file
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

    # Test 6: 2D tensor for shape verification
    create_tar_pytorch_file(
        "test_data/tar_2d_tensor.tar",
        {
            "matrix": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 4]),
        },
        {"matrix": "FloatStorage"},
    )

    print("\nAll TAR format test files created!")
    print("\nTo run tests: cargo test -p burn-store --features pytorch test_tar")


if __name__ == "__main__":
    main()
