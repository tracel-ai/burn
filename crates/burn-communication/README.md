# Burn Communication

> Client/server networking used by [Burn](https://github.com/tracel-ai/burn)'s remote backend

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-communication.svg)](https://crates.io/crates/burn-communication)
[![Documentation](https://docs.rs/burn-communication/badge.svg)](https://docs.rs/burn-communication)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

`Protocol` abstracts a transport with a `ProtocolServer` that routes connections to handlers and a
`ProtocolClient` that opens `CommunicationChannel`s to an `Address`.

- `websocket` (feature `websocket`): a WebSocket implementation of `Protocol`.
- `external_comm` (feature `data-service`): lets one server download a tensor directly from
  another, without routing the data through the client.

This crate is an implementation detail of [burn-remote](https://github.com/tracel-ai/burn/tree/main/crates/burn-remote); applications do not use it
directly.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
