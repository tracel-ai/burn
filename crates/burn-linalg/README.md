# Burn Linear Algebra

> Linear algebra operations for [Burn](https://github.com/tracel-ai/burn) tensors

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-linalg.svg)](https://crates.io/crates/burn-linalg)
[![Documentation](https://docs.rs/burn-linalg/badge.svg)](https://docs.rs/burn-linalg)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- Decompositions: `lu`, `qr` and `svd`.
- `det`, `trace`, `diag`, `outer`, `matvec` and `cosine_similarity`.
- Vector and matrix norms: L0, L1, L2, Lp, max and min absolute value, and `vector_normalize`.

## Usage

Enable both `linalg` and a backend feature on `burn`, and use the operations through
`burn::linalg`:

```toml
burn = { version = "0.22", features = ["linalg", "flex"] }
```

No execution backend is enabled by default when depending on this crate directly; select `flex`,
`wgpu` or another backend feature on it. Enabling only `burn/flex` does not enable this crate.

Linear algebra previously lived at `burn_tensor::linalg`; that path is removed in 0.22.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
