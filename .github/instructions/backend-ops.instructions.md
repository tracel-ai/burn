---
applyTo: "crates/burn-backend/**,crates/burn-autodiff/**,crates/burn-fusion/**,crates/burn-cubecl-fusion/**,crates/burn-router/**,crates/burn-ir/**,crates/burn-dispatch/**,crates/burn-cubecl/**,crates/burn-flex/**"
---

# Backend Operations

An operation reaches a backend through several layers, and each one must handle it. See the
[operation guide](../../contributor-book/src/guides/adding-a-new-operation-to-burn.md).

- Backend trait methods should be required, not `unimplemented!()` defaults. A default panics for
  every backend that relies on it, and under fusion the panic surfaces at the next flush rather than
  at the call.
- Decorator backends (autodiff, fusion, router, dispatch) must override new operations and
  capability flags. Without an override they inherit the default decomposition, so a dedicated
  kernel or backward pass is silently skipped.
- New operations in `burn-ir` should reuse existing IR types and the `impl_ir_create!` macro where
  the shape and semantics match, and represent scalars with `ScalarIr`.
- A fusion change that turns a decomposed operation into an opaque one stops neighboring operations
  from fusing with it. Note the trade-off, or teach `burn-cubecl-fusion` the new operation.
- Autodiff should checkpoint only the tensors whose gradients are tracked, and must not assume both
  operands require gradients.
- Kernels should take the element type from the tensor dtype instead of hardcoding `f32`, and keep
  accumulation precision separate from storage precision. They should handle strided, permuted,
  broadcast, and zero-sized layouts, or reject them explicitly.
- Prefer existing CubeCL and CubeK utilities (`LinearView`, `FastDivmod`, tensor strides) over
  manual index math or extra stride tensors.
- Results must agree across Flex and the CubeCL backends, including edge cases such as empty
  windows, `-inf` padding, NaN propagation, and tie-breaking for max and argmax.
