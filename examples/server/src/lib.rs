use burn::{
    server::{Channel, IrohChannel, IrohChannelBuilder, IrohRelays, RemoteSecret, TokenAuthorizer},
    tensor::Device,
};

/// Host `Device::default()` for remote clients.
///
/// `REMOTE_BACKEND_TRANSPORT` picks how: `websocket` by default, on port `REMOTE_BACKEND_PORT` or
/// 3000, or `iroh`, configured as [`iroh_channel`] describes.
pub fn start() {
    let channel = match std::env::var("REMOTE_BACKEND_TRANSPORT").as_deref() {
        Err(_) | Ok("websocket") => {
            let port = port().unwrap_or(3000);
            println!("listening on websocket port {port}");
            Channel::WebSocket { port }
        }
        Ok("iroh") => {
            let channel = iroh_channel();
            println!("listening on iroh as {}", channel.id());
            Channel::Iroh { channel }
        }
        Ok(other) => panic!("REMOTE_BACKEND_TRANSPORT is websocket or iroh, got {other}"),
    };

    burn::server::start(Device::default(), channel);
}

/// An Iroh channel from the environment:
/// - `REMOTE_BACKEND_TOKEN`, required: the token every client must present.
/// - `REMOTE_BACKEND_KEY`: the server's identity file, `target/remote-backend.key` by default.
/// - `REMOTE_BACKEND_RELAYS`: `public` by default, `disabled`, or the URL of a relay you run.
/// - `REMOTE_BACKEND_PORT`: the UDP port to bind, required when relays are disabled.
pub fn iroh_channel() -> IrohChannel {
    let token = std::env::var("REMOTE_BACKEND_TOKEN")
        .ok()
        .and_then(TokenAuthorizer::new)
        .expect("REMOTE_BACKEND_TOKEN holds the non-empty token clients present");
    let key = std::env::var("REMOTE_BACKEND_KEY")
        .unwrap_or_else(|_| "target/remote-backend.key".to_string());
    if let Some(parent) = std::path::Path::new(&key).parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("{} can hold the identity file: {err}", parent.display()));
    }
    let secret = RemoteSecret::load_or_create(&key)
        .unwrap_or_else(|err| panic!("{key} holds, or can hold, the server's identity: {err}"));
    let relays: IrohRelays =
        std::env::var("REMOTE_BACKEND_RELAYS").map_or(IrohRelays::Public, |relays| {
            relays
                .parse()
                .unwrap_or_else(|err| panic!("REMOTE_BACKEND_RELAYS: {err}"))
        });

    let channel = IrohChannelBuilder::new(secret)
        .with_relays(relays.clone())
        .with_authorizer(token);
    match (port(), relays) {
        (Some(port), _) => channel.with_port(port),
        (None, IrohRelays::Disabled) => {
            panic!("REMOTE_BACKEND_PORT names the UDP port clients dial when relays are disabled")
        }
        (None, _) => channel,
    }
    .build()
}

fn port() -> Option<u16> {
    std::env::var("REMOTE_BACKEND_PORT")
        .ok()
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
}
