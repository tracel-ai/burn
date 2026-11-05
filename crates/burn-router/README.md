# Burn Router

> Records [Burn](https://github.com/tracel-ai/burn) tensor operations as IR and forwards them to wherever they execute

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-router.svg)](https://crates.io/crates/burn-router)
[![Documentation](https://docs.rs/burn-router/badge.svg)](https://docs.rs/burn-router)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`BackendRouter` is a backend whose operations are not run in place. Each one is described as
[burn-ir](https://github.com/tracel-ai/burn/tree/main/crates/burn-ir) and sent through a `RouterChannel` to a client; on the receiving side, a
`TensorInterpreter` replays the operations on a real backend.

This is the layer under remote execution ([burn-remote](https://github.com/tracel-ai/burn/tree/main/crates/burn-remote) sends the operations over
the network) and graph capture ([burn-capture](https://github.com/tracel-ai/burn/tree/main/crates/burn-capture) records them without executing).
Operations defined by backend extensions travel through a `CustomOpRegistry`.

Applications do not use this crate directly.

## Feature Flags

- `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
- `fusion`: fuse routed operations before they are sent.
- `tracing`: instrument operations with the `tracing` crate.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
