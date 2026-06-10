# Burn Pack

> The **burnpack** binary serialization format for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-pack.svg)](https://crates.io/crates/burn-pack)
[![Documentation](https://docs.rs/burn-pack/badge.svg)](https://docs.rs/burn-pack)

`burn-pack` is a small, dependency-light crate that reads and writes the burnpack container
format. It is **tensor-library-agnostic**: it depends only on
[`burn-std`](https://crates.io/crates/burn-std) (for `DType` / `Bytes`), `serde`, and a CBOR
codec — no `burn-core`, no `burn-tensor`. It knows the on-disk format but has no notion of Burn
modules or tensors.

Higher layers bridge between burnpack and richer types:

- [`burn-core`](https://crates.io/crates/burn-core) converts its `TensorSnapshot`s to/from
  `burn_pack::Tensor` and exposes the `Record` API.
- [`burn-store`](https://crates.io/crates/burn-store) provides `BurnpackStore` along with
  PyTorch/SafeTensors interop.

If you just want to save and load Burn models, use `burn-core` / `burn-store`. Reach for
`burn-pack` directly when you need to produce or consume the raw format (tooling, converters,
other frameworks, embedded readers).

## Features

- **Minimal dependencies** — `burn-std` + `serde` + CBOR only; no tensor crates
- **Lazy file loading** — reading from a file backs each tensor with [`Bytes::from_file`], so data
  is read (and DMA'd to the GPU) only when accessed; 256-byte aligned tensor data keeps it efficient
- **`Bytes`-native** — `Tensor` entries hold a [`Bytes`] buffer directly, integrating with the rest
  of the Burn ecosystem (shared/static/file-backed allocations)
- **ParamId persistence** — optional per-tensor id for stateful training round-trips
- **Hardened reader** — magic/version checks plus DoS limits on metadata size, tensor count,
  tensor size, CBOR recursion depth, and file size
- **No-std support** — the format core works in embedded and WASM environments (file I/O is behind
  the `std` feature)

## Quick Start

```rust
use burn_pack::{Bytes, DType, Reader, Tensor, Writer};

// A 2x2 f32 tensor, as raw little-endian bytes.
let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();

let tensor = Tensor::new(
    "weight".to_string(),
    DType::F32,
    vec![2, 2],
    Some(42), // optional param id
    Bytes::from_bytes_vec(raw),
);

// Write to an in-memory buffer ...
let packed = Writer::new(vec![tensor])
    .with_metadata("producer", "my-tool")
    .to_bytes()
    .unwrap();

// ... and read it back.
let reader = Reader::from_bytes(packed).unwrap();
let tensors = reader.get_tensors().unwrap();

assert_eq!(tensors[0].name, "weight");
assert_eq!(tensors[0].shape.to_vec(), vec![2, 2]);
assert_eq!(tensors[0].param_id, Some(42));
assert_eq!(reader.metadata()["producer"], "my-tool");
```

### Files

```rust,ignore
use burn_pack::{Reader, Writer};

// Stream directly to disk (memory-efficient for large models).
Writer::new(tensors).write_to_file("model.bpk").unwrap();

// Read the header + metadata up front; each tensor's data is backed by `Bytes::from_file`
// and read lazily when accessed (and DMA'd to the GPU efficiently).
let reader = Reader::from_file("model.bpk").unwrap();
let tensors = reader.get_tensors().unwrap();
```

Reading from a file (`from_file`) keeps tensor data lazy. Reading from an in-memory buffer
(`from_bytes`) copies each tensor's bytes out of the buffer.

## File Format

All multi-byte integers are little-endian.

```text
┌──────────────────────────────────────────────────────────────┐
│ Header — 10 bytes (HEADER_SIZE)                              │
│   magic         : u32  — 0x4255524E "BURN" (MAGIC_NUMBER)    │
│   version       : u16  — format version (FORMAT_VERSION)     │
│   metadata_size : u32  — byte length of the CBOR metadata    │
├──────────────────────────────────────────────────────────────┤
│ Metadata — CBOR, `metadata_size` bytes (Metadata)           │
│   tensors : map<name, TensorDescriptor>                     │
│     dtype        : DType                                     │
│     shape        : list<u64>                                │
│     data_offsets : (start, end)  relative to the data section│
│     param_id     : optional u64  (training-state identity)  │
│   metadata : map<string, string>  user key/value pairs      │
├──────────────────────────────────────────────────────────────┤
│ Padding to the next 256-byte boundary                       │
│   (aligned_data_section_start)                              │
├──────────────────────────────────────────────────────────────┤
│ Tensor data section                                         │
│   each tensor's bytes start on a 256-byte boundary          │
│   (TENSOR_ALIGNMENT) so the file can be mmap'd and tensors   │
│   sliced zero-copy.                                          │
└──────────────────────────────────────────────────────────────┘
```

**Why 256-byte alignment?** It lets a reader mmap the file and hand out tensor slices without
copying, while satisfying the alignment requirements of every element type (including 8-byte
`f64`), cache lines, and GPU coalesced access. 256 bytes matches the choice made by GGUF, MLX,
ncnn, and other major formats.

### Safety limits

Reading is hardened against malicious or corrupt inputs. The reader rejects files that exceed any
of `MAX_METADATA_SIZE`, `MAX_TENSOR_COUNT`, `MAX_TENSOR_SIZE`, `MAX_CBOR_RECURSION_DEPTH`, or (for
the file loaders) `MAX_FILE_SIZE`, and validates that the file is large enough to contain every
tensor it claims — returning `Error::ValidationError` otherwise.

## Inspecting a pack

For a file-backed reader, `get_tensors` does not read any tensor data until a tensor's `bytes`
are accessed — so you can inspect dtype/shape/param-id for free:

```rust,ignore
let reader = Reader::from_file("model.bpk")?;

for t in reader.get_tensors()? {
    println!("{}: {:?} {:?}", t.name, t.dtype, t.shape); // no file read yet
}

// User key/value metadata, and a single tensor's raw bytes by name.
let producer = &reader.metadata()["producer"];
let raw = reader.tensor_data("encoder.weight")?;
```

## Feature Flags

| Feature   | Default | Description                                              |
| --------- | ------- | -------------------------------------------------------- |
| `std`     | yes     | File I/O (`Reader::from_file` / `Writer::write_to_file`) |

Disable defaults for no-std targets:

```toml
[dependencies]
burn-pack = { version = "0.22", default-features = false }
```

## License

This project is dual-licensed under MIT and Apache-2.0.
