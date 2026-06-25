# Burn Remote

Burn Remote executes tensor operations on compute peers reached through
[Iroh](https://iroh.computer/). Iroh is the primary transport: peers are identified by
cryptographic endpoint IDs, direct connections are preferred, and configured relays are used
when NAT traversal cannot establish a direct path.

## Client

Applications own the Iroh endpoint configuration. This keeps identity persistence, relay policy,
address lookup, and fleet discovery outside Burn:

```rust,ignore
use burn::{Device, Tensor};
use burn::backend::remote::{Endpoint, RemoteNode, endpoint::presets};

let endpoint = Endpoint::builder(presets::N0)
    // .secret_key(persistent_secret_key)
    // .relay_mode(custom_relay_mode)
    // .address_lookup(platform_lookup)
    .bind()
    .await?;
let node = RemoteNode::from_endpoint(endpoint);

// Supplied by your platform, invitation, or other discovery mechanism.
let compute_peer = fleet.lookup("gpu-worker-7").await?;
let device = Device::remote_iroh(&node, compute_peer, 0);

let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
```

`RemoteNode` is process-level. Clone it rather than creating one per device: clones share one
Iroh endpoint, executor, peer connection pool, and multiplexed QUIC connections.

Platforms can issue a `RemoteTicket` containing an `EndpointAddr` and opaque authorization bytes.
Burn passes the credential to the compute peer's `PeerAuthorizer`; signature format, expiry,
tenant policy, and fleet membership remain application concerns.

## Compute peer

```rust,ignore
use burn::{Device, server::{self, Channel}};
use burn::backend::remote::RemoteNode;

let node = RemoteNode::bind().await?;
println!("compute peer: {}", node.endpoint().addr());

server::start_async(
    Device::cuda(0),
    Channel::Iroh { node },
).await;
```

For an endpoint shared with other Iroh protocols, register Burn's composable handler in the
application router:

```rust,ignore
use burn_remote::BURN_REMOTE_ALPN;
use iroh::protocol::Router;

let burn = node
    .protocol::<MyBackend>(devices)
    .with_authorizer(|request| platform.verify(request.peer, request.credential));

let router = Router::builder(node.endpoint().clone())
    .accept(BURN_REMOTE_ALPN, burn)
    .accept(MY_OTHER_ALPN, other_protocol)
    .spawn();
```

## Tensor movement

Moving a tensor between different Iroh compute peers does not route the payload through the
client. The destination peer opens an authenticated stream directly to the source peer. Each
transfer uses a random, short-lived capability bound to the destination's authenticated endpoint
identity and limited to the number of downloads requested by the operation.

Multiple devices hosted by the same compute peer retain the in-process fast path.
Tensor movement between an Iroh peer and a legacy WebSocket peer is not supported.

## WebSocket compatibility

The `websocket` feature preserves `Device::remote("ws://host:port", index)` and
`Channel::WebSocket`. It is intended for compatibility; new integrations should use Iroh.
