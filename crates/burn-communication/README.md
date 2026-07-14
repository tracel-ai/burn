# Burn Communication

Abstractions for network communication

The Protocol trait defines how to communicate in a server/client style.
The server can set up routes with callbacks upon connection.

## WebSocket

Communication with WebSockets is implemented with the `websocket` feature.

## Tensor Data Service

The tensor data service provides easy utilities to share tensors peer-to-peer.
One peer can expose a tensor, and another can download it. Each peer is both a client and a server.
