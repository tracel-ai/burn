# burn-einsum

Dependency-free equation parsing shared by Burn's `einsum!` macro and runtime
einsum API. The crate supports `no_std` with `alloc`.

Equations support ASCII labels (`A-Z`, then `a-z`), repeated input labels,
ellipsis, scalar operands, and explicit or implicit output. Parsing validates
equation syntax without tensor shapes. Operand ranks, diagonal sizes, and
broadcast compatibility are checked by the tensor executor.
