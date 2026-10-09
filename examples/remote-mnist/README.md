# Remote MNIST

Train the [MNIST example](../mnist)'s model on a GPU in another machine, then run inference with it
there. The model, data pipeline and training loop are the MNIST example's own; this example only
connects to a remote device and runs them on it.

The example shows you how to:

- Reach a server's device over Iroh or WebSocket.
- Train with burn-train on that device, metrics included.
- Load the trained model and run inference on the same device.

The GPU machine runs the [server example](../server) with its backend feature (`cuda`, `rocm`,
`vulkan`, or `webgpu` by default). The client can run anywhere.

## Iroh

Iroh connects two machines by identity rather than by address, end-to-end encrypted, and gets
through NATs and firewalls. The server only serves clients that present its token.

Pick a token on the GPU machine and start the server:

```bash
export REMOTE_BACKEND_TOKEN=$(openssl rand -hex 32)
REMOTE_BACKEND_TRANSPORT=iroh cargo run -p server --example server --release --features cuda
```

It prints its id, `listening on iroh as <id>`. The id comes from a key the server creates in
`target/remote-backend.key` on its first start, or wherever `REMOTE_BACKEND_KEY` points, so it
stays the same across restarts. Keep that file private: whoever holds it can pose as the server.

On the client, with the same token:

```bash
export REMOTE_BACKEND_TOKEN=<the server's token>
cargo run -p remote-mnist --example remote-mnist --release -- train <id>
cargo run -p remote-mnist --example remote-mnist --release -- infer <id>
```

### Relays

A relay forwards traffic between peers that cannot reach each other directly. It only sees
encrypted packets, and peers switch to a direct connection whenever they can. Both ends must use the
same setting.

| Server `REMOTE_BACKEND_RELAYS` | Client flags | What happens |
|---|---|---|
| `public` (default) | none | n0's public relays, and n0's lookup finds the server by its id. |
| `https://relay.example.com` | `--relays https://relay.example.com` | A relay you run, such as [iroh-relay](https://github.com/n0-computer/iroh/tree/main/iroh-relay); nothing goes through n0. |
| `disabled`, with `REMOTE_BACKEND_PORT=4433` | `--relays disabled --address gpu-host:4433` | Direct only; the client must be able to reach that UDP port. |

## WebSocket

WebSocket has no encryption: anyone who can read the traffic sees the tensors, and the token if
there is one. Use it only on a network you trust. A token is optional here, and keeps stray clients
on that network off the GPU: set `REMOTE_BACKEND_TOKEN` on both machines as for Iroh.

On the GPU machine, listening on port 3000 unless `REMOTE_BACKEND_PORT` says otherwise:

```bash
cargo run -p server --example server --release --features cuda
```

On the client:

```bash
cargo run -p remote-mnist --example remote-mnist --release -- train ws://gpu-host:3000
cargo run -p remote-mnist --example remote-mnist --release -- infer ws://gpu-host:3000
```

## Where the model goes

`train` saves the model on the client, under `/tmp/burn-example-mnist`. `infer` loads it from there,
sends the weights to the server, and classifies 1000 test images on its GPU.
