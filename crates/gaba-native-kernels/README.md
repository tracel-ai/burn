# gaba-native-kernels

This crate provides small, opt-in native kernels for Gaba Burn.

Contract
- Function: `gemm_f32(a, b, c, m, n, k)`
- ABI: C-compatible, row-major matrices; `a` is m x k, `b` is k x n, `c` is m x n.
- Types: f32 inputs and outputs for this prototype.

Build
- By default the crate uses a pure-Rust fallback implementation (`gemm_rust`).
- To enable the Zig native kernel, build with the `zig` Cargo feature. When enabled,
  `build.rs` will invoke `zig` to compile `native/gemm.zig` into a dynamic library and
  link it automatically.

Example

```bash
# run tests with default Rust kernel
cargo test -p gaba-native-kernels

# build and run with Zig native kernel (requires `zig` in PATH)
cargo run -p gaba-native-kernels --example quick_bench --features zig
```

Notes
- Kernels are feature-gated and opt-in. CI may provide an additional job that installs Zig
  to run benchmarks and regression tracking.
