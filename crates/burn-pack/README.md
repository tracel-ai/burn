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
- **Zero-copy loading** — 256-byte aligned tensor data enables memory-mapped, copy-free slicing
- **Lazy materialization** — `Tensor` entries carry metadata plus a closure that yields the raw
  bytes only when asked
- **ParamId persistence** — optional per-tensor id for stateful training round-trips
- **Hardened reader** — magic/version checks plus DoS limits on metadata size, tensor count,
  tensor size, CBOR recursion depth, and file size
- **No-std support** — the format core works in embedded and WASM environments (file I/O and
  mmap are behind the `std` / `memmap` features)

## Quick Start

```rust
use burn_pack::{Bytes, DType, Reader, Tensor, Writer};

// A 2x2 f32 tensor, as raw little-endian bytes.
let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();

let tensor = Tensor::from_bytes(
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
assert_eq!(tensors[0].shape, vec![2, 2]);
assert_eq!(tensors[0].param_id, Some(42));
assert_eq!(reader.metadata().metadata["producer"], "my-tool");
```

### Files and memory mapping

```rust,ignore
use burn_pack::{Reader, Writer};

// Stream directly to disk (memory-efficient for large models).
Writer::new(tensors).write_to_file("model.bpk").unwrap();

// Load via mmap (default with the `memmap` feature); slice tensors zero-copy.
let reader = Reader::from_file("model.bpk").unwrap();
let tensors = reader.get_tensors_zero_copy(true).unwrap();
```

Zero-copy slicing requires shared-backed bytes (an mmap'd file, or `Bytes::from_shared` /
`Bytes::from_static`). Slicing an in-memory `Bytes::from_bytes_vec` buffer is intentionally
unsupported and falls back to copying via `get_tensors()`.

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

`Reader` exposes the metadata without materializing any tensor data:

```rust,ignore
let reader = Reader::from_file("model.bpk")?;

for name in reader.tensor_names() {
    let d = &reader.metadata().tensors[name];
    println!("{name}: {:?} {:?}", d.dtype, d.shape);
}

// Read a single tensor's raw bytes by name.
let raw = reader.tensor_data("encoder.weight")?;
```

## Feature Flags

| Feature   | Default | Description                                              |
| --------- | ------- | -------------------------------------------------------- |
| `std`     | yes     | File I/O and other std-only functionality                |
| `memmap`  | yes     | Memory-mapped, zero-copy file loading (implies `std`)    |

Disable defaults for no-std targets:

```toml
[dependencies]
burn-pack = { version = "0.22", default-features = false }
```

## License

This project is dual-licensed under MIT and Apache-2.0.
