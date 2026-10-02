# Burn Remote

Burn Remote runs tensor operations on the devices a server hosts, in another process on this machine
or another. A client sends the operations, the server runs them, and only what the client reads
comes back.

It has two transports:

- **Iroh**, the default, works across any network, authenticated and encrypted. A server is named
  by a cryptographic id; direct connections are preferred, and relays carry the traffic when NAT
  traversal cannot establish a direct path.
- **WebSocket**, with the `remote-websocket` feature, is the simplest setup on a trusted network:
  the same machine, a LAN, containers or CI. It is unencrypted.

## Client

Describe the server with a `RemoteHost`, then connect one of its devices:

```rust,ignore
use burn::remote::RemoteHost;
use burn::tensor::{Device, Tensor};

let host = RemoteHost::iroh(server_id).with_credential(token);
let device = Device::remote_options(&host).init()?;

let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
```

- `init_async().await` connects from async code, and is the only form in a browser.
- `.device_index(1)` picks another of the server's devices.
- `Device::enumerate(DeviceType::Remote(host))` lists all of them, beside any local device type, and
  each connects on first use; `host.devices()` does the same and returns an error where `enumerate`
  panics.
- `RemoteHost::websocket("ws://gpu:3000")` reaches a WebSocket server.

An Iroh server's relays must match the client's:
`RemoteHost::iroh(IrohHost::new(server_id).with_relays(relays))`. With relays disabled,
`.with_address(address)` gives each address the server listens on.

An application that already runs an Iroh endpoint, for its own identity, address lookup or other
protocols, dials from it with `IrohHost::with_endpoint`. That endpoint keeps its own settings; the
ones Burn binds send no segmentation-offloaded (GSO) batches because of
[iroh#4555](https://github.com/n0-computer/iroh/issues/4555).

A device whose session ended, as when its server restarted, is not reopened: connecting again gives
a new device, and the old one's tensors are gone. Each new device keeps a thread for the life of the
process. A server that went away without closing the session is noticed once the transport gives up
on it, and until then connecting returns the old device.

## Server

```rust,ignore
use burn::server::{IrohIdentity, IrohTransport, RemoteServer, TokenAuthorizer};
use burn::tensor::Device;

let transport = IrohTransport::new(IrohIdentity::load_or_create("server.key")?);
println!("server id: {}", transport.id());

RemoteServer::new([Device::cuda(0)])
    .with_authorizer(TokenAuthorizer::new(token)?)
    .serve(transport)?;
```

- A client picks a device by its position in the server's list.
- `serve` blocks until Ctrl+C or `SIGTERM`. `serve_async(transport).await` runs until its future is
  dropped, which also ends the live sessions.
- The transport uses n0's public relays by default. `.with_relays(IrohRelays::Private { url })`
  goes through a relay you run instead, and `.with_relays(IrohRelays::Disabled).with_port(4433)`
  serves direct connections only, on a UDP port clients dial.
- `WebSocketTransport::new(3000)` serves over WebSocket. The authorizer applies to both transports;
  over WebSocket, a token stops stray clients on a trusted network, not someone reading the traffic.
- `with_custom_op::<B, _>` hosts a backend extension's operations. A backend outside Burn's own serves
  through `burn_remote::server::BackendServer<B>`.

For an endpoint shared with other Iroh protocols, register Burn's handler in the application's
router:

```rust,ignore
use burn::server::{AuthorizationRequest, BURN_REMOTE_ALPN, RemoteServer};
use iroh::protocol::Router;

let burn = RemoteServer::new([Device::cuda(0)])
    .with_authorizer(|request: AuthorizationRequest<'_>| {
        platform.verify(request.client, request.credential)
    })
    .into_protocol(&endpoint)?;

let router = Router::builder(endpoint)
    .accept(BURN_REMOTE_ALPN, burn)
    .accept(MY_OTHER_ALPN, other_protocol)
    .spawn();
```

## Tensor movement

Moving a tensor between different Iroh servers does not route the payload through the client. The
destination server opens an authenticated stream directly to the source server. Each transfer uses
a random, short-lived capability bound to the destination's authenticated endpoint identity and
limited to the number of downloads requested by the operation. These streams do not pass through
the authorizer: the capability is what the client's session granted.

Multiple devices hosted by the same server retain the in-process fast path. Tensor movement between
an Iroh server and a WebSocket server is not supported.
