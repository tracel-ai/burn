#![cfg(all(feature = "client", feature = "server", feature = "iroh"))]

use burn_flex::Flex;
use burn_ir::BackendIr;
use burn_remote::{
    BURN_REMOTE_ALPN, RemoteDevice,
    server::{AllowAll, IrohRemoteProtocol},
    telemetry::TelemetryProbe,
};
use burn_tensor::{Device, Tensor};
use iroh::{
    Endpoint, EndpointAddr, RelayMode, address_lookup::MemoryLookup, endpoint::presets,
    protocol::Router,
};
use std::{panic, sync::mpsc, thread, time::Duration};

/// Past the first retries, well inside the retry window.
const ADDRESS_LATE_BY: std::time::Duration = std::time::Duration::from_millis(700);

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
    let protocol = IrohRemoteProtocol::<B>::new(
        endpoint.clone(),
        vec![Default::default()],
        std::sync::Arc::new(authorizer),
        probe,
        burn_remote::server::CustomOpRegistry::default(),
    );
    Router::builder(endpoint)
        .accept(BURN_REMOTE_ALPN, protocol)
        .spawn()
}

/// Far beyond what a test here takes when it works, so only a hang reaches it.
const HANG_LIMIT: Duration = Duration::from_secs(30);

const TOKIO_COOP_BUDGET: usize = 128;

/// A blocking hang cannot be cancelled, so this fails after [`HANG_LIMIT`] and leaks the stuck
/// thread.
fn within_hang_limit(test: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let name = thread::current().name().unwrap_or("test").to_string();
    let thread = thread::Builder::new()
        .name(name)
        .spawn(move || {
            test();
            let _ = done.send(());
        })
        .unwrap();
    if let Err(mpsc::RecvTimeoutError::Timeout) = finished.recv_timeout(HANG_LIMIT) {
        panic!("still blocked after {HANG_LIMIT:?}");
    }
    if let Err(payload) = thread.join() {
        panic::resume_unwind(payload);
    }
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
async fn a_dial_waits_for_an_iroh_address_published_late() {
    let server = local_endpoint().await;
    let router = spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled());
    let lookup = MemoryLookup::new();
    let client = Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Disabled)
        .clear_ip_transports()
        .bind_addr("127.0.0.1:0")
        .unwrap()
        .address_lookup(lookup.clone())
        .bind()
        .await
        .unwrap();
    let address = server.addr();
    tokio::spawn(async move {
        tokio::time::sleep(ADDRESS_LATE_BY).await;
        lookup.add_endpoint_info(address);
    });

    let remote = RemoteDevice::iroh(&client, EndpointAddr::new(server.id()), 0);
    remote.connect();
    let device = Device::new(remote);

    let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
    assert_eq!(output.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);

    router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "no address lookup is configured")]
async fn a_dial_with_no_address_and_no_lookup_is_not_retried() {
    let server = local_endpoint().await;
    let _router = spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled());
    let client = local_endpoint().await;

    RemoteDevice::iroh(&client, EndpointAddr::new(server.id()), 0).connect();
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

#[test]
fn blocking_reads_inside_a_tokio_task_outlast_its_budget() {
    within_hang_limit(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let server = local_endpoint().await;
            let client = local_endpoint().await;
            let router = spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled());

            let remote = RemoteDevice::iroh(&client, server.addr(), 0);
            remote.connect();
            let device = Device::new(remote);
            for _ in 0..=TOKIO_COOP_BUDGET {
                let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
                assert_eq!(
                    output.try_into_vec_as::<f32>().unwrap(),
                    vec![2.0, 4.0, 6.0]
                );
            }

            router.shutdown().await.unwrap();
        });
    });
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
