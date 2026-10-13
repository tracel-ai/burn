# burn-einsum

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-einsum.svg)](https://crates.io/crates/burn-einsum)
[![Documentation](https://docs.rs/burn-einsum/badge.svg)](https://docs.rs/burn-einsum)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Dependency-free equation parsing and execution planning shared by Burn's `einsum!`
macro and runtime einsum API. The crate supports `no_std` with `alloc`.

Equations support ASCII labels (`A-Z`, then `a-z`), repeated input labels,
ellipsis, scalar operands, and explicit or implicit output. Parsing validates
equation syntax without tensor shapes. `Equation::plan()` determines diagonal
extraction, canonical axis alignment, last-use reductions, and left-to-right
contraction groups and permutations. Ellipsis dimensions remain symbolic so the
same plan supports different operand ranks.

The macro generates tensor operations from this plan at compile time. The
runtime API interprets the same plan. Operand ranks, diagonal sizes, broadcasting,
and shape-dependent matrix multiplication details are checked by the tensor
executor.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
