# Copilot Code Review

Read [AGENTS.md](../AGENTS.md) for the repository layout, architecture, and conventions. These notes
cover how to review a pull request in this repository.

## Priorities

Report correctness problems first, then missing test coverage, then documentation that contradicts
the code. Each finding should name a concrete input or configuration that fails and what happens.
Skip a finding you cannot back with the code in this checkout.

Look closely for:

- A new or changed backend operation that is not forwarded through every layer. Autodiff, fusion,
  router, dispatch, and IR fall back to default decompositions or capability flags (for example
  `has_ctc_loss_backward`) unless they override them, so the change silently does not take effect.
- `unimplemented!()` or `todo!()` defaults on trait methods, or panics reachable from public APIs on
  normal inputs, such as quantized (`QFloat`) tensors, zero-sized dimensions, or unsupported dtypes.
- Hardcoded `f32` or `i64` where the dtype should come from the input. Epsilons should come from the
  dtype (`FloatDType::finfo`). Some devices, such as wgpu, lack 64-bit integers.
- Numerical regressions: overflow in low precision, lost higher-precision accumulation, NaN or
  infinity handling, and division by zero on empty inputs.
- Host synchronization (`into_data`, `into_scalar`, value checks) in forward passes, losses, and
  metrics where tensor operations would do.
- User input validated deep in a backend instead of at the `burn-tensor` API boundary with
  `check!(TensorCheck::...)`. Panic messages should name the operation and the offending values.
- Behavior changes to existing public APIs: argument order, output shapes, default features, and
  removed items not called out in the PR description.
- Copy-paste errors between similar code paths, such as a real/imaginary or input/output mix-up, or
  a benchmark that measures a different operation than its name says.
- Doc comments, examples, and test names that no longer match the code they describe.
- `Cargo.lock` changes that downgrade unrelated crates. Changes to the `windows*` crates break
  Windows CI.

## Tests

- Tests should check values, not only shapes. Inputs should be able to tell a correct implementation
  from a plausible wrong one: avoid all-ones or symmetric data, `arange` inputs where indices equal
  values, and configurations where two modes produce the same result.
- A bug fix needs a regression test that fails without the fix. New validation needs a
  `#[should_panic(expected = ...)]` test.
- Behavior shared across backends belongs in `burn-backend-tests`, gated by backend features rather
  than enabling a backend unconditionally.

## Avoid

- Flagging missing `alloc` imports, unused imports, or type errors. CI builds every supported
  configuration, including `no_std`; report a compile error only if you can point to the target that
  fails.
- Asking a focused PR to fix pre-existing issues or refactor surrounding code. If a pre-existing
  issue is serious, mention it once as out of scope.
- Speculative failures that need malformed internal data or invariants the code already guarantees.
- Performance suggestions without a clear hot path or a cost the change introduces. `clone()` on
  tensors and `to_contiguous()` on contiguous tensors are cheap.
- Replacing `unwrap()` in examples and tests, or style preferences already handled by rustfmt and
  clippy.
- Repeating the same finding on every occurrence. Report it once and list the other locations.

Keep each comment short: the problem, a failing case, and a suggested fix when it is clear.
