pub use super::*;

mod like;
mod matmul;
mod quantize;

// The `extended` suite is only enabled for backends with native (non-packed) quantized
// storage AND a complete set of quantized ops. Today that means Flex.
//
// - cube backends are excluded: PackedU32 storage requires the last dim to be a multiple of
//   the pack factor (4 int8s per u32), which most of these test shapes violate, so the
//   quantized tensors can't even be constructed (`q_from_data` panics with "Can't store in u32").
//
// - tch is excluded too. Its quantization primitives (`q_from_data`, `quantize`, `dequantize`,
//   ...) are `unimplemented!()`, so running `extended` against it would fail. (It also
//   doesn't enable the `quantization` feature, so it isn't selected here in the first place.)
//
// - remote runs it only when `flex` is enabled too, and then passes only against a server whose
//   backend stores quantized values natively, for the cube reason above.
//
// - autodiff is excluded for a different reason: its `QTensorOps` impl delegates every method to
//   the inner backend, so `Autodiff<Flex>` quantizes fine. It simply never reaches this module,
//   because the autodiff suite is a separate test target (`tests/autodiff.rs`) that does not
//   include `tests/tensor/`.
//
// Enabling `flex` here means the `extended` suite now also runs under the `tensor_f16` target.
// That f16 path is why a couple of `maxmin` tests use a slightly
// looser tolerance: reductions like `min_dim` re-quantize their output, so a value is rounded to
// int8 twice (input quantization, then re-quantization of the reduced result). For small-magnitude
// values this accumulated rounding lands a hair over the tight `rel_abs(2e-2, 1e-2)` bound at f16
// (e.g. `1.0` -> ~0.97998, rel error 2.00e-2), whereas f32's finer scale representation keeps the
// same value just inside it (~0.9802, rel error 1.98e-2). It is quantization noise, not a logic
// error, so those cases use `rel_abs(2e-2, 3e-2)`.
#[cfg(feature = "flex")]
mod extended;
