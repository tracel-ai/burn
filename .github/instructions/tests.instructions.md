---
applyTo: "crates/burn-backend-tests/**,crates/**/tests/**"
---

# Tests

Follow the [testing guide](../../contributor-book/src/getting-started/testing.md).

- Assert the behavior under test. Numerical operations need value checks. If a change only affects
  shapes, metadata, or input validation, test that behavior directly. A test that cannot fail is a
  finding.
- Keep tests focused on behavior introduced or changed by the PR. Each case should catch a distinct
  plausible failure. Extend existing tests where practical, and avoid redundant cases, assertions
  that mirror the implementation, and unrelated edge cases. A test that would break under a
  behavior-preserving refactor is asserting implementation.
- Test each contract once, at the layer that owns it. Tensor operation results belong in
  `burn-backend-tests`; backend-local tests (as in `burn-flex`) cover only backend internals such as
  dtype storage selection, and `burn-nn` modules test their own composition rather than re-testing
  the functional op. A regression test at the owning layer is enough.
- Do not widen visibility or add production code that only tests call. Test through the public or
  backend API instead.
- Choose inputs that tell a correct implementation from a plausible wrong one: avoid all-ones or
  symmetric data, `arange` inputs where indices equal values, a single batch and channel, and
  configurations where the modes being compared give the same result (for example, padding 0 when
  comparing `count_include_pad`, or odd sizes when comparing ceil and floor mode).
- Cover the edge cases the change actually touches, such as zero-sized dimensions,
  non-power-of-two sizes, dims other than 0 or the last, duplicate indices, quantized inputs, and
  gradients when only one operand requires them.
- Reference values should come from an independent source such as PyTorch, not from the
  implementation under test. A round trip through the same implementation, such as `rfft` then
  `irfft`, can miss errors that cancel out.
- Prefer named tests over loops over configurations, so a failure identifies its case. The
  `TestCase` pattern (for example `AvgPool1dTestCase`) shares the comparison logic. A test name
  should not promise more than its input exercises.
- `#[should_panic(expected = ...)]` should match the specific message of the check being tested, so
  an earlier unrelated panic does not make it pass.
- Comparisons across backends need tolerances suited to the dtype and device. Byte-for-byte checks
  need inputs whose results are exactly representable.
- Gate backend-specific tests with backend features instead of enabling a backend unconditionally.
  The `fusion/` suite also runs on Flex, so tests that assert CubeCL fusion decisions need
  `#[cfg(feature = "cube")]`, as in `fusion/mod.rs`.

Before calling a test redundant, check that it does not independently guard a public API, a file
format, parity with a reference implementation, a numeric tolerance, or a specific backend.
