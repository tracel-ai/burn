pub use super::*;

mod matmul;
mod quantize;

// The `extended` suite is only enabled for backends with native (non-packed) quantized
// storage AND a complete set of quantized ops. Today that means ndarray and flex.
//
// - cube backends are excluded: PackedU32 storage requires the last dim to be a multiple of
//   the pack factor (4 int8s per u32), which most of these test shapes violate, so the
//   quantized tensors can't even be constructed (`q_from_data` panics with "Can't store in u32").
//
// - candle, router, tch and autodiff are excluded too. We dropped their `unimplemented!()`
//   overrides for `q_gather`/`q_select`/`q_slice`/`q_expand` so they fall back to the default
//   `dequantize -> float op -> quantize` path, but that does NOT make them functional: their
//   other quantized methods are largely unimplemented. The quantization primitives themselves
//   (`q_from_data`, `quantize`, `dequantize`, ...) are still `unimplemented!()`/`todo!()`, so the
//   fallback simply moves the panic into `dequantize`. Running `extended` against any of these
//   backends would fail. (They also don't enable the `quantization` feature, so they aren't
//   selected here in the first place.)
//
// Enabling `flex` here means the `extended` suite now also runs under the `tensor_f16` target.
// That f16 path is why a couple of `maxmin` tests use a slightly
// looser tolerance: reductions like `min_dim` re-quantize their output, so a value is rounded to
// int8 twice (input quantization, then re-quantization of the reduced result). For small-magnitude
// values this accumulated rounding lands a hair over the tight `rel_abs(2e-2, 1e-2)` bound at f16
// (e.g. `1.0` -> ~0.97998, rel error 2.00e-2), whereas f32's finer scale representation keeps the
// same value just inside it (~0.9802, rel error 1.98e-2). It is quantization noise, not a logic
// error, so those cases use `rel_abs(2e-2, 3e-2)`.
#[cfg(any(feature = "ndarray", feature = "flex"))]
mod extended;
