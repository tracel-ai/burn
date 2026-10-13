---
applyTo: "**"
---

# Design and Scope

Besides checking that a change works, check whether it belongs in the codebase in this form. Raise
these points as questions with supporting evidence, such as the existing item a change duplicates or
the crate that owns the concept. Maintainers decide.

- Placement. Code should live in the crate and layer that owns the concept. Functional tensor
  operations belong in `burn-tensor` (for example `burn_tensor::module::attention`), linear algebra
  in `burn-linalg`, and signal processing in `burn-signal`, rather than being reimplemented from
  basic tensor operations in `burn-nn`, `burn-train`, or an example. A requirement that only one
  crate needs should not be added to the `Backend` trait or the public `Tensor` API. Behavior that
  belongs to a dependency's type, such as one from CubeCL, should be fixed upstream.
- Reuse. Before accepting a new function, trait, or IR type, look for an existing one with the same
  semantics and point to it. Near-copies of existing code drift apart, so they should share an
  implementation.
- Public surface. Each new public item, trait, and re-export should have a user-facing need. Prefer
  crate-private helpers, plain functions over traits with a single implementation, and methods on
  the type that owns the state. Avoid blanket extension traits on standard types.
- Root causes. A workaround for a bug in another layer, such as fusion, a backend, or CubeCL, should
  be a fix in that layer or a linked issue rather than a local special case. A fallback such as a
  round trip through the host should be deliberate and documented on the public API with its cost,
  as `burn_linalg::svd` does. An undocumented fallback that hides an unsupported case can be worse
  than an explicit unsupported error.
- Scope. Changes unrelated to the PR's stated goal should be split out: dependency bumps, test
  device or feature changes, edits to other crates, and drive-by refactors. Large features and new
  public concepts should have an agreed design in an issue first.
- Dependencies. New crates and feature flags need a justification. Prefer what the workspace
  already uses.
- Speculative generality. Configuration options, generic parameters, traits, and extension points
  need a current caller.
- Leftovers. Repro tests without assertions, benchmarks in the test suite, enabled debug logging,
  commented-out code, local configuration files, and stale TODOs should not be merged.
