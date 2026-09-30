# Burn Remote

Burn Remote executes tensor operations on compute peers reached through
[Iroh](https://iroh.computer/). Iroh is the primary transport: peers are identified by
cryptographic endpoint IDs, direct connections are preferred, and configured relays are used
when NAT traversal cannot establish a direct path.

## Client

Describe the server with an `IrohPeer`: its id, the relays it uses, and the credential its
authorizer checks. Burn binds the endpoint and dials every device of the peer over one connection:

```rust,ignore
use burn::remote::IrohPeerBuilder;
use burn::tensor::{Device, Tensor};

let peer = IrohPeerBuilder::new(server_id).credential(token).build();
let device = Device::remote_iroh_peer(&peer, 0).await?;

let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
```

The peer's relays must match the server's. With relays disabled, `.address(...)` gives each address
the server listens on.

An application that manages identity, address lookup or discovery itself passes its own Iroh
endpoint to `Device::remote_iroh` instead. That endpoint keeps Iroh's settings, including
segmentation offload, which [iroh#4555](https://github.com/n0-computer/iroh/issues/4555) makes
worth turning off:

```rust,ignore
use burn::remote::Endpoint;
use iroh::endpoint::{QuicTransportConfig, presets};

let transport = QuicTransportConfig::builder()
    .enable_segmentation_offload(false)
    .build();
let endpoint = Endpoint::builder(presets::N0)
    .transport_config(transport)
    .bind()
    .await?;
let device = Device::remote_iroh(&endpoint, compute_peer, 0);
```

## Compute peer

```rust,ignore
use burn::server::{self, Channel, IrohChannelBuilder, RemoteSecret, TokenAuthorizer};
use burn::tensor::Device;

let secret = RemoteSecret::load_or_create("server.key")?;
let channel = IrohChannelBuilder::new(secret)
    .authorizer(TokenAuthorizer::new(token).expect("A non-empty token"))
    .build();
println!("compute peer: {}", channel.id());

server::start_async(Device::cuda(0), Channel::Iroh { channel }).await;
```

The channel uses n0's public relays by default. `.relays(IrohRelays::Private { url })` goes through
a relay you run instead, and `.relays(IrohRelays::Disabled).port(4433)` serves direct connections
only, on a UDP port clients dial. `IrohRelays` and `RelayUrl` come from `burn::server`.

For an endpoint shared with other Iroh protocols, register Burn's composable handler in the
application router:

```rust,ignore
use burn::server::{self, BURN_REMOTE_ALPN};
use iroh::protocol::Router;

let burn = server::protocol(Device::cuda(0), &endpoint)
    .authorizer(|request| platform.verify(request.peer, request.credential))
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

The `websocket` feature preserves `Device::remote("ws://host:port", index)` and
`Channel::WebSocket`. It is intended for compatibility; new integrations should use Iroh.
