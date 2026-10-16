---
applyTo: "crates/burn-backend/**,crates/burn-autodiff/**,crates/burn-fusion/**,crates/burn-cubecl-fusion/**,crates/burn-router/**,crates/burn-ir/**,crates/burn-dispatch/**,crates/burn-cubecl/**,crates/burn-flex/**"
---

# Backend Operations

An operation reaches a backend through several layers, and each one should be checked. See the
[operation guide](../../contributor-book/src/guides/adding-a-new-operation-to-burn.md).

- Backend trait methods should be required, not `unimplemented!()` defaults. A default panics for
  every backend that relies on it, and under fusion the panic surfaces at the next flush rather than
  at the call.
- When a decorator backend (autodiff, fusion, router, dispatch) inherits a default, check whether
  the default loses required behavior or intended acceleration, such as a dedicated kernel, a
  backward pass, or a capability flag like `has_ctc_loss_backward`. Some defaults are intentional:
  fusion keeps `linear` decomposed so its matmul and add can fuse with neighboring operations.
- New operations in `burn-ir` should reuse existing IR types and the `impl_ir_create!` macro where
  the shape and semantics match, and represent scalars with `ScalarIr`.
- A fusion change that turns a decomposed operation into an opaque one stops neighboring operations
  from fusing with it. Note the trade-off, or teach `burn-cubecl-fusion` the new operation.
- Autodiff should checkpoint only the state that a tracked operand's backward needs, and must not
  assume both operands require gradients. For `mul`, the LHS gradient needs the RHS, so the RHS is
  checkpointed only when the LHS is tracked.
- Kernels should take the element type from the tensor dtype instead of hardcoding `f32`, and keep
  accumulation precision separate from storage precision. They should handle strided, permuted,
  broadcast, and zero-sized layouts, or reject them explicitly.
- Prefer existing CubeCL and CubeK utilities (`LinearView`, `FastDivmod`, tensor strides) over
  manual index math or extra stride tensors.
- Results must agree across Flex and the CubeCL backends, including edge cases such as empty
  windows, `-inf` padding, NaN propagation, and tie-breaking for max and argmax.
