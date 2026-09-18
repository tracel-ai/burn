#![cfg(all(feature = "client", feature = "server", feature = "iroh"))]

use burn_flex::Flex;
use burn_ir::BackendIr;
use burn_remote::{
    BURN_REMOTE_ALPN, RemoteDevice,
    server::{AllowAll, IrohRemoteProtocol},
    telemetry::TelemetryProbe,
};
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

fn spawn_router<B: BackendIr>(
    endpoint: Endpoint,
    authorizer: impl burn_remote::server::PeerAuthorizer,
    probe: TelemetryProbe,
) -> Router {
    spawn_router_hosting::<B>(endpoint, 1, authorizer, probe)
}

fn spawn_router_hosting<B: BackendIr>(
    endpoint: Endpoint,
    devices: usize,
    authorizer: impl burn_remote::server::PeerAuthorizer,
    probe: TelemetryProbe,
) -> Router {
    let protocol = IrohRemoteProtocol::<B>::new(
        endpoint.clone(),
        vec![Default::default(); devices],
        std::sync::Arc::new(authorizer),
        probe,
        burn_remote::server::CustomOpRegistry::default(),
    );
    Router::builder(endpoint)
        .accept(BURN_REMOTE_ALPN, protocol)
        .spawn()
}

#[tokio::test(flavor = "multi_thread")]
async fn executes_over_iroh_session_stream() {
    let server = local_endpoint().await;
    let client = local_endpoint().await;
    let router = spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled());

    let remote = RemoteDevice::iroh(&client, server.addr(), 0);
    remote.connect();
    let device = Device::new(remote);

    let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
    assert_eq!(
        output.try_into_vec_as::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );

    router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn transfers_tensor_directly_between_iroh_compute_peers() {
    let source_server = local_endpoint().await;
    let target_server = local_endpoint().await;
    let client = local_endpoint().await;

    let source_router =
        spawn_router::<Flex>(source_server.clone(), AllowAll, TelemetryProbe::disabled());
    let target_router =
        spawn_router::<Flex>(target_server.clone(), AllowAll, TelemetryProbe::disabled());

    let source_remote = RemoteDevice::iroh(&client, source_server.addr(), 0);
    let target_remote = RemoteDevice::iroh(&client, target_server.addr(), 0);
    source_remote.connect();
    target_remote.connect();
    let source = Device::new(source_remote);
    let target = Device::new(target_remote);

    let tensor = Tensor::<1>::from_floats([3.0, 5.0, 7.0], &source);
    let transferred = tensor.to_device(&target);
    assert_eq!(
        transferred.try_into_vec_as::<f32>().unwrap(),
        vec![3.0, 5.0, 7.0]
    );

    source_router.shutdown().await.unwrap();
    target_router.shutdown().await.unwrap();
}

/// The synchronous client path used by scripts, REPLs and Rust notebooks: no `async`, no ambient
/// runtime in the calling code. The device is created on the client's runtime (so the session
/// reuses it, the way [`RemoteNode::bind_blocking`] does internally) and every operation is then
/// driven synchronously off it.
#[test]
fn synchronous_client_round_trip() {
    // Server on its own local runtime, kept alive for the duration of the test.
    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let server_guard = server_runtime.enter();
    let server = server_runtime.block_on(local_endpoint());
    let router = spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled());
    let server_addr = server.addr();
    drop(server_guard);

    // Client on a node that owns its runtime, used entirely synchronously from this (non-runtime)
    // thread, exactly what a notebook cell does.
    let client_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let client_endpoint = client_runtime.block_on(local_endpoint());

    // Create the device on the client's runtime so the session captures it; the round-trip below
    // then runs from this non-runtime thread, exactly what a notebook cell does.
    let remote = {
        let _guard = client_runtime.enter();
        RemoteDevice::iroh(&client_endpoint, server_addr, 0)
    };
    remote.connect();
    let device = Device::new(remote);

    let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
    assert_eq!(
        output.try_into_vec_as::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );

    server_runtime.block_on(router.shutdown()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn passes_application_credentials_to_the_peer_authorizer() {
    let server = local_endpoint().await;
    let client = local_endpoint().await;
    let router = spawn_router::<Flex>(
        server.clone(),
        |request: burn_remote::server::AuthorizationRequest<'_>| {
            (request.credential == b"fleet-ticket")
                .then_some(())
                .ok_or_else(|| "invalid fleet ticket".to_string())
        },
        TelemetryProbe::disabled(),
    );
    let remote = RemoteDevice::iroh_authorized(&client, server.addr(), 0, b"fleet-ticket".to_vec());
    remote.connect();
    let device = Device::new(remote);
    let data = Tensor::<1>::from_floats([4.0], &device).to_data();
    assert_eq!(data.try_into_vec::<f32>().unwrap(), vec![4.0]);

    router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[cfg(feature = "fusion")]
async fn fused_compute_surfaces_as_graph_telemetry() {
    use burn_remote::telemetry::{TelemetryEvent, TelemetryProbe, TrafficAggregator};
    use std::time::Duration;

    let server = local_endpoint().await;
    let client = local_endpoint().await;

    let (probe, mut events) = TelemetryProbe::channel(4096);
    let router = spawn_router::<Flex>(server.clone(), AllowAll, probe);
    let remote = RemoteDevice::iroh(&client, server.addr(), 0);
    remote.connect();
    let device = Device::new(remote);

    // A multi-op float expression fuses into a cached graph; running it twice forces a replay, and
    // each read flushes the fusion stream so the server actually executes the graph.
    for _ in 0..2 {
        let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
        let y = ((x * 2.0) + 1.0).exp().log();
        let _ = y.to_data();
    }

    // Fold the stream the same way a logger or dashboard would, and check the derived economics.
    let mut aggregator = TrafficAggregator::default();
    let (mut saw_registered, mut saw_executed) = (false, false);
    let collect = async {
        while !(saw_registered && saw_executed && aggregator.snapshot().fused_ops > 0) {
            let Some(event) = events.recv().await else {
                break;
            };
            aggregator.apply(&event);
            match event.as_ref() {
                TelemetryEvent::GraphRegistered { ops, bytes, .. } => {
                    saw_registered = !ops.is_empty() && *bytes > 0
                }
                TelemetryEvent::GraphExecuted { .. } => saw_executed = true,
                _ => {}
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), collect)
        .await
        .expect("fused-path telemetry did not arrive in time");

    assert!(
        saw_registered,
        "expected a GraphRegistered event carrying the graph's ops and size"
    );
    assert!(saw_executed, "expected a GraphExecuted replay heartbeat");
    assert!(
        aggregator.snapshot().fused_ops > 0,
        "the aggregator should price the replayed graph's ops as fused"
    );

    router.shutdown().await.unwrap();
}

/// Pipeline stages alternating between two servers, the shape a split model runs in: every hop
/// crosses servers and the client reads nothing until the end, so transfers between the same pair of
/// servers are in flight in both directions at once.
#[tokio::test(flavor = "multi_thread")]
async fn stages_alternating_between_two_servers_transfer_both_ways() {
    let first = local_endpoint().await;
    let second = local_endpoint().await;
    let client = local_endpoint().await;
    let routers = [
        spawn_router_hosting::<Flex>(first.clone(), 2, AllowAll, TelemetryProbe::disabled()),
        spawn_router_hosting::<Flex>(second.clone(), 2, AllowAll, TelemetryProbe::disabled()),
    ];

    let stages: Vec<Device> = [
        (first.addr(), 0),
        (second.addr(), 0),
        (first.addr(), 1),
        (second.addr(), 1),
    ]
    .into_iter()
    .map(|(peer, index)| {
        let remote = RemoteDevice::iroh(&client, peer, index);
        remote.connect();
        Device::new(remote)
    })
    .collect();

    // A stalled transfer parks a worker deep in the backend, where it cannot be killed, so the test
    // reports from here and the process exit carries the worker away.
    let (done, waiting) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut tensor = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &stages[0]);
        for stage in &stages[1..] {
            tensor = tensor.to_device(stage) * 2.0;
        }
        let _ = done.send(tensor.into_data().try_into_vec::<f32>().unwrap());
    });
    let values = waiting
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the stages to finish");
    assert_eq!(values, vec![8.0, 16.0, 24.0, 32.0]);

    for router in routers {
        router.shutdown().await.unwrap();
    }
}

/// The target downloads from the shared endpoint down the connection that endpoint's client
/// dialed, before the endpoint hosted its server.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_that_is_also_a_client_is_downloaded_from() {
    let shared = local_endpoint().await;
    let target = local_endpoint().await;
    let client = local_endpoint().await;
    let target_router = spawn_router::<Flex>(target.clone(), AllowAll, TelemetryProbe::disabled());

    let shared_as_client = RemoteDevice::iroh(&shared, target.addr(), 0);
    shared_as_client.connect();
    let shared_router = spawn_router::<Flex>(shared.clone(), AllowAll, TelemetryProbe::disabled());

    let source = RemoteDevice::iroh(&client, shared.addr(), 0);
    let destination = RemoteDevice::iroh(&client, target.addr(), 0);
    source.connect();
    destination.connect();
    let (source, destination) = (Device::new(source), Device::new(destination));

    let (done, waiting) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tensor = Tensor::<1>::from_floats([3.0, 5.0, 7.0], &source).to_device(&destination);
        let _ = done.send(tensor.try_into_vec_as::<f32>().unwrap());
    });
    let values = waiting
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the transfer to finish");
    assert_eq!(values, vec![3.0, 5.0, 7.0]);

    shared_router.shutdown().await.unwrap();
    target_router.shutdown().await.unwrap();
}
