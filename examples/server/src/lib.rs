use burn::{
    server::{
        IrohIdentity, IrohRelays, IrohTransport, RemoteServer, TokenAuthorizer, Transport,
        WebSocketTransport,
    },
    tensor::{Device, DeviceType},
};

/// Host the default backend's devices for remote clients, every one of them when the backend
/// lists several, so a client can drive them with data-parallel training.
///
/// `REMOTE_BACKEND_TRANSPORT` picks how: `websocket` by default, on port `REMOTE_BACKEND_PORT` or
/// 3000, or `iroh`, configured as [`iroh_transport`] describes. `REMOTE_BACKEND_TOKEN` is the
/// token every client must present, required over Iroh.
pub fn start() {
    let token = std::env::var("REMOTE_BACKEND_TOKEN").ok().map(|token| {
        TokenAuthorizer::new(token).expect("REMOTE_BACKEND_TOKEN holds a non-empty token")
    });
    let transport: Transport = match std::env::var("REMOTE_BACKEND_TRANSPORT").as_deref() {
        Err(_) | Ok("websocket") => {
            let port = port().unwrap_or(3000);
            println!("listening on websocket port {port}");
            WebSocketTransport::new(port).into()
        }
        Ok("iroh") => {
            assert!(
                token.is_some(),
                "REMOTE_BACKEND_TOKEN holds the token every Iroh client presents"
            );
            let transport = iroh_transport();
            println!("listening on iroh as {}", transport.id());
            transport.into()
        }
        Ok(other) => panic!("REMOTE_BACKEND_TRANSPORT is websocket or iroh, got {other}"),
    };

    let server = RemoteServer::new(hosted_devices());
    let server = match token {
        Some(token) => server.with_authorizer(token),
        None => server,
    };
    if let Err(err) = server.serve(transport) {
        panic!("the server stopped: {err}");
    }
}

/// Every device of the type `Device::default()` is, or that device alone when its type lists one
/// or cannot list its hardware.
fn hosted_devices() -> Vec<Device> {
    let kinds: Vec<DeviceType> = vec![
        #[cfg(feature = "cuda")]
        DeviceType::Cuda,
        #[cfg(feature = "rocm")]
        DeviceType::Rocm,
        #[cfg(feature = "vulkan")]
        DeviceType::Vulkan,
        #[cfg(feature = "webgpu")]
        DeviceType::WebGpu,
        #[cfg(feature = "flex")]
        DeviceType::Flex,
    ];

    let wgpu_kinds: Vec<DeviceType> = vec![
        #[cfg(feature = "vulkan")]
        DeviceType::Vulkan,
        #[cfg(feature = "webgpu")]
        DeviceType::WebGpu,
    ];
    let listed = |kinds: Vec<DeviceType>| {
        kinds
            .into_iter()
            .map(|kind| Device::enumerate(kind).into_vec())
    };

    let default = Device::default();
    listed(kinds)
        .find(|devices| devices.contains(&default))
        // The wgpu default names whichever adapter it lands on, so no list contains it.
        .or_else(|| listed(wgpu_kinds).find(|devices| !devices.is_empty()))
        .filter(|devices| devices.len() > 1)
        .unwrap_or_else(|| vec![default])
}

/// An Iroh transport from the environment:
/// - `REMOTE_BACKEND_KEY`: the server's identity file, `target/remote-backend.key` by default.
/// - `REMOTE_BACKEND_RELAYS`: `public` by default, `disabled`, or the URL of a relay you run.
/// - `REMOTE_BACKEND_PORT`: the UDP port to bind, required when relays are disabled.
pub fn iroh_transport() -> IrohTransport {
    let key = std::env::var("REMOTE_BACKEND_KEY")
        .unwrap_or_else(|_| "target/remote-backend.key".to_string());
    if let Some(parent) = std::path::Path::new(&key).parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("{} can hold the identity file: {err}", parent.display()));
    }
    let identity = IrohIdentity::load_or_create(&key)
        .unwrap_or_else(|err| panic!("{key} holds, or can hold, the server's identity: {err}"));
    let relays: IrohRelays =
        std::env::var("REMOTE_BACKEND_RELAYS").map_or(IrohRelays::Public, |relays| {
            relays
                .parse()
                .unwrap_or_else(|err| panic!("REMOTE_BACKEND_RELAYS: {err}"))
        });

    let transport = IrohTransport::new(identity).with_relays(relays.clone());
    match (port(), relays) {
        (Some(port), _) => transport.with_port(port),
        (None, IrohRelays::Disabled) => {
            panic!("REMOTE_BACKEND_PORT names the UDP port clients dial when relays are disabled")
        }
        (None, _) => transport,
    }
}

fn port() -> Option<u16> {
    std::env::var("REMOTE_BACKEND_PORT")
        .ok()
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
}
