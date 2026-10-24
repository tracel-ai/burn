---
applyTo: "**"
---

# Design and Scope

Besides checking that a change works, check whether it belongs in the codebase in this form. A
design finding needs no failing input; back it with concrete complexity, duplication, or API
friction, such as the existing item a change duplicates, and explain the benefit of the proposed
change. Ask questions when intent or trade-offs remain unclear. Maintainers decide.

- Placement. Reuse existing functional operations (for example
  `burn_tensor::module::attention`) instead of reimplementing them. Place new operations according
  to their intended users and scope; application-specific compositions need not become core tensor
  APIs. Before extending `Backend` or `Tensor`, check whether the operation warrants a shared
  contract or belongs in an extension crate such as `burn-linalg` or `burn-signal`. Behavior that
  belongs to a dependency's type, such as one from CubeCL, should be fixed upstream.
- Reuse. Before accepting a new function, trait, or IR type, look for an existing one with the same
  semantics and point to it. Near-copies of existing code drift apart, so they should share an
  implementation.
- Abstractions and public surface. Inspect actual callers when evaluating new abstractions and
  public APIs. Check that names, signatures, visibility, and required imports follow existing
  conventions, and that new types represent meaningful state or behavior rather than merely grouping
  helpers. Each new public item and re-export should have a user-facing need.
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
