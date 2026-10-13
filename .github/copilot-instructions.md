# Copilot Code Review

See [AGENTS.md](../AGENTS.md) for layout, architecture, and conventions. These notes cover review.

## Priorities

Report correctness problems first, then missing test coverage, then documentation that contradicts
the code. Each finding should name a concrete input or configuration that fails and what happens.
Skip a finding you cannot back with the code in this checkout.

Look closely for:

- A backend operation whose dedicated implementation is lost in a decorator layer. Autodiff,
  fusion, router, and dispatch inherit default decompositions and capability flags (for example
  `has_ctc_loss_backward`) unless they override them. Some defaults are intentional, such as fusion
  keeping operations decomposed so they fuse with neighbors; flag a default only when it loses
  required behavior or intended acceleration.
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

- Tests should assert the behavior under test: numerical operations need value checks, not only
  shapes. Inputs should be able to tell a correct implementation from a plausible wrong one: avoid
  all-ones or symmetric data, `arange` inputs where indices equal values, and configurations where
  two modes produce the same result.
- Redundant or low-value tests are a finding too: cases that catch the same failure, assertions
  that mirror the implementation, and edge cases unrelated to the change. Tests that guard a public
  API, a file format, or parity with a reference are not redundant.
- A bug fix needs a regression test that fails without the fix. New validation needs a
  `#[should_panic(expected = ...)]` test.
- Behavior shared across backends belongs in `burn-backend-tests`, gated by backend features.

## Avoid

- Flagging missing `alloc` imports, unused imports, or type errors. CI builds every supported
  configuration, including `no_std`; report a compile error only if you can point to the target that
  fails.
- Asking a focused PR to fix pre-existing issues or refactor surrounding code. If a pre-existing
  issue is serious, mention it once as out of scope.
- Speculative failures that need malformed internal data or invariants the code already guarantees.
- Performance suggestions without a clear hot path or a cost the change introduces. Cloning a
  tensor is cheap because it shares the underlying buffer.
- Replacing `unwrap()` in examples and tests, or style preferences already handled by rustfmt and
  clippy.
- Repeating a finding on every occurrence. Report it once and list the other locations.

Keep comments short: the problem, a failing case, and a fix when it is clear.
