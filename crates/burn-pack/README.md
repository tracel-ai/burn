# Burn Pack

> The **burnpack** binary serialization format for the Burn deep learning framework

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-pack.svg)](https://crates.io/crates/burn-pack)
[![Documentation](https://docs.rs/burn-pack/badge.svg)](https://docs.rs/burn-pack)

`burn-pack` reads and writes the burnpack container format. It is tensor-library-agnostic and
dependency-light: it depends only on [`burn-std`](https://crates.io/crates/burn-std) (for `DType`
/ `Bytes`), `serde`, and a CBOR codec — it knows the on-disk format but has no notion of Burn
modules. Tensor data is `Bytes`-native and read lazily from files (256-byte aligned for
zero-copy mmap and efficient GPU transfers), and the reader is hardened against malformed input.

If you just want to save and load Burn models, use the higher-level
[`burn-core`](https://crates.io/crates/burn-core) record API or
[`burn-store`](https://crates.io/crates/burn-store) (which adds PyTorch/SafeTensors interop).
Reach for `burn-pack` directly only to produce or consume the raw format.

## Usage

```rust
use burn_pack::{Bytes, DType, Reader, Tensor, Writer};

let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
let tensor = Tensor::new("weight".to_string(), DType::F32, vec![2, 2], None, Bytes::from_bytes_vec(raw));

let packed = Writer::new(vec![tensor]).into_bytes().unwrap();
let reader = Reader::from_bytes(packed).unwrap();
assert_eq!(reader.into_tensors().unwrap()[0].shape.to_vec(), vec![2, 2]);
```

Use `Writer::write_to_file` / `Reader::from_file` for disk I/O (the default `std` feature; disable
it for no-std targets). See the [docs](https://docs.rs/burn-pack) for the format layout and the
full API.

## License

This project is dual-licensed under MIT and Apache-2.0.
