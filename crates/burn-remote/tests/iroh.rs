#![cfg(all(feature = "client", feature = "server", feature = "iroh"))]

use burn_flex::Flex;
use burn_remote::{BURN_REMOTE_ALPN, RemoteNode};
use burn_tensor::{Device, Tensor};
use iroh::{Endpoint, RelayMode, endpoint::presets, protocol::Router};

async fn local_endpoint() -> Endpoint {
    Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Disabled)
        .clear_ip_transports()
        .bind_addr("127.0.0.1:0")
        .unwrap()
        .bind()
        .await
        .unwrap()
}

async fn local_node() -> RemoteNode {
    RemoteNode::from_endpoint(local_endpoint().await)
}

#[tokio::test(flavor = "multi_thread")]
async fn executes_over_iroh_session_stream() {
    let server = local_node().await;
    let client = local_node().await;
    let router = server.protocol::<Flex>(vec![Default::default()]).serve();

    let remote = client.device(server.endpoint().addr(), 0);
    remote.connect();
    let device = Device::new(remote);

    let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
    assert_eq!(
        output.to_data().to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );

    router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn transfers_tensor_directly_between_iroh_compute_peers() {
    let source_server = local_node().await;
    let target_server = local_node().await;
    let client = local_node().await;

    let source_router = source_server
        .protocol::<Flex>(vec![Default::default()])
        .serve();
    let target_router = target_server
        .protocol::<Flex>(vec![Default::default()])
        .serve();

    let source_remote = client.device(source_server.endpoint().addr(), 0);
    let target_remote = client.device(target_server.endpoint().addr(), 0);
    source_remote.connect();
    target_remote.connect();
    let source = Device::new(source_remote);
    let target = Device::new(target_remote);

    let tensor = Tensor::<1>::from_floats([3.0, 5.0, 7.0], &source);
    let transferred = tensor.to_device(&target);
    assert_eq!(
        transferred.to_data().to_vec::<f32>().unwrap(),
        vec![3.0, 5.0, 7.0]
    );

    source_router.shutdown().await.unwrap();
    target_router.shutdown().await.unwrap();
}

/// The synchronous client path used by scripts, REPLs and Rust notebooks: no `async`, no ambient
/// runtime in the calling code. The client node owns its runtime (via an explicit handle here, the
/// way [`RemoteNode::bind_blocking`] does internally) and every operation is driven synchronously.
#[test]
fn synchronous_client_round_trip() {
    // Server on its own local runtime, kept alive for the duration of the test.
    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server_guard = server_runtime.enter();
    let server = server_runtime.block_on(local_node());
    let router = server.protocol::<Flex>(vec![Default::default()]).serve();
    let server_addr = server.endpoint().addr();
    drop(server_guard);

    // Client on a node that owns its runtime, used entirely synchronously from this (non-runtime)
    // thread — exactly what a notebook cell does.
    let client_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let client_endpoint = client_runtime.block_on(local_endpoint());
    let client = RemoteNode::from_endpoint_on(client_endpoint, client_runtime.handle().clone());

    let remote = client.device(server_addr, 0);
    remote.connect();
    let device = Device::new(remote);

    let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
    assert_eq!(
        output.to_data().to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );

    server_runtime.block_on(router.shutdown()).unwrap();
}

/// `bind_blocking` builds its own runtime and binds without an ambient one.
#[test]
fn bind_blocking_needs_no_ambient_runtime() {
    let node = RemoteNode::bind_blocking().expect("bind_blocking should bind an endpoint");
    let _ = node.id();
}

#[tokio::test(flavor = "multi_thread")]
async fn passes_application_credentials_to_the_peer_authorizer() {
    let server = local_node().await;
    let client = local_node().await;
    let protocol = server
        .protocol::<Flex>(vec![Default::default()])
        .with_authorizer(|request: burn_remote::server::AuthorizationRequest<'_>| {
            (request.credential == b"fleet-ticket")
                .then_some(())
                .ok_or_else(|| "invalid fleet ticket".to_string())
        });
    let router = Router::builder(server.endpoint().clone())
        .accept(BURN_REMOTE_ALPN, protocol)
        .spawn();

    let remote = client.device_authorized(server.endpoint().addr(), 0, b"fleet-ticket".to_vec());
    remote.connect();
    let device = Device::new(remote);
    let data = Tensor::<1>::from_floats([4.0], &device).to_data();
    assert_eq!(data.to_vec::<f32>().unwrap(), vec![4.0]);

    router.shutdown().await.unwrap();
}
