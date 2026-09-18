# Burn Remote

Burn Remote executes tensor operations on compute peers reached through
[Iroh](https://iroh.computer/). Iroh is the primary transport: peers are identified by
cryptographic endpoint IDs, direct connections are preferred, and configured relays are used
when NAT traversal cannot establish a direct path.

## Client

Applications own the Iroh endpoint configuration. This keeps identity persistence, relay policy,
address lookup, and fleet discovery outside Burn:

```rust,ignore
use burn::tensor::{Device, Tensor};
use iroh::{Endpoint, endpoint::presets};

let endpoint = Endpoint::builder(presets::N0)
    // .secret_key(persistent_secret_key)
    // .relay_mode(custom_relay_mode)
    // .address_lookup(platform_lookup)
    .bind()
    .await?;

// Supplied by your platform, invitation, or other discovery mechanism.
let compute_peer = fleet.lookup("gpu-worker-7").await?;
let device = Device::remote_iroh(&endpoint, compute_peer, 0);

let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
```

Build every device from the same endpoint. It is the client's identity, which is what a compute
peer authorizes, and the devices built from it share one connection per compute peer. The same
endpoint can also host a server, registered as below.

`Device::remote_iroh_authorized` also sends an opaque credential, which Burn passes to the compute
peer's `PeerAuthorizer`. Signature format, expiry, tenant policy, and fleet membership remain
application concerns.

## Compute peer

```rust,ignore
use burn::{server::{self, Channel, RemoteSecret}, tensor::Device};

let secret = RemoteSecret::random();
println!("compute peer: {}", secret.id());

server::start_async(
    Device::cuda(0),
    Channel::Iroh { secret: Box::new(secret) },
).await;
```

This serves every peer that dials it. To authorize peers, or to share an endpoint with other Iroh
protocols, register Burn's composable handler in the application router:

```rust,ignore
use burn::{
    server::{self, AuthorizationRequest, BURN_REMOTE_ALPN},
    tensor::Device,
};
use iroh::protocol::Router;

let burn = server::protocol(Device::cuda(0), &endpoint)
    .with_authorizer(|request: AuthorizationRequest<'_>| {
        platform.verify(request.peer, request.credential)
    })
    .build();

let router = Router::builder(endpoint)
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

The `websocket` feature preserves `Device::remote_websocket("ws://host:port", index)` and
`Channel::WebSocket`. It is intended for compatibility; new integrations should use Iroh.
