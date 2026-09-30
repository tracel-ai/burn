#![cfg(all(feature = "client", feature = "server", feature = "iroh"))]

use burn_flex::Flex;
use burn_ir::BackendIr;
use burn_remote::{
    BURN_REMOTE_ALPN, RemoteDevice,
    server::{AllowAll, IrohRemoteProtocol},
    telemetry::{TelemetryEvent, TelemetryProbe},
};
use burn_tensor::{DType, Device, Int, Tensor, TensorData};
use iroh::{
    Endpoint, EndpointAddr, RelayMode, address_lookup::MemoryLookup, endpoint::presets,
    protocol::Router,
};
use std::{panic, sync::mpsc, thread, time::Duration};
use tokio::task::coop;

/// Past the first retries, well inside the retry window.
const ADDRESS_LATE_BY: Duration = Duration::from_millis(700);

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
async fn a_client_that_disconnects_without_closing_ends_its_session() {
    let server = local_endpoint().await;
    let client = local_endpoint().await;
    let (probe, mut events) = TelemetryProbe::channel(4096);
    let router = spawn_router::<Flex>(server.clone(), AllowAll, probe);

    let remote = RemoteDevice::iroh(&client, server.addr(), 0);
    remote.connect();
    let device = Device::new(remote);
    let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
    output.try_into_vec_as::<f32>().unwrap();

    client.close().await;
    let session_closed = async {
        while let Some(event) = events.recv().await {
            if let TelemetryEvent::SessionClosed { .. } = event.as_ref() {
                return true;
            }
        }
        false
    };
    let closed = tokio::time::timeout(Duration::from_secs(10), session_closed).await;
    assert!(
        matches!(closed, Ok(true)),
        "the server kept the session of a client that disconnected"
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
fn unsigned_int_uploads_read_back_and_cast() {
    within_hang_limit(|| {
        let server_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = server_runtime.block_on(local_endpoint());
        let router = {
            let _guard = server_runtime.enter();
            spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled())
        };
        let client_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let client = client_runtime.block_on(local_endpoint());
        let remote = {
            let _guard = client_runtime.enter();
            RemoteDevice::iroh(&client, server.addr(), 0)
        };
        remote.connect();
        let device = Device::new(remote);

        let pixels = TensorData::new(vec![0u8, 7, 128, 255], [2, 2]);
        let pixels = Tensor::<2, Int>::from_data(pixels, (&device, DType::U8));
        assert_eq!(
            pixels.clone().try_into_vec_as::<u8>().unwrap(),
            vec![0, 7, 128, 255]
        );
        assert_eq!(
            pixels.float().try_into_vec_as::<f32>().unwrap(),
            vec![0.0, 7.0, 128.0, 255.0]
        );

        server_runtime.block_on(router.shutdown()).unwrap();
    });
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
            while coop::has_budget_remaining() {
                coop::consume_budget().await;
            }
            let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
            assert_eq!(
                output.try_into_vec_as::<f32>().unwrap(),
                vec![2.0, 4.0, 6.0]
            );

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

#[cfg(feature = "fusion")]
mod loader_uploads {
    use super::*;
    use burn_remote::telemetry::{DrainStatus, OpClass, TelemetryEvent};
    use burn_tensor::{Int, TensorData};
    use std::collections::HashSet;
    use std::sync::mpsc;

    const STEPS: usize = 8;
    const UPLOADS_PER_BATCH: usize = 2;

    struct Batch {
        images: Tensor<2>,
        targets: Tensor<1, Int>,
    }

    impl Batch {
        fn new(step: usize, device: &Device) -> Self {
            Self {
                images: Tensor::from_data(TensorData::new(vec![step as f32; 12], [3, 4]), device),
                targets: Tensor::from_data(TensorData::new(vec![step as i64; 3], [3]), device),
            }
        }
    }

    fn assert_server_drops_every_upload(consume: fn(Batch)) {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(local_endpoint());
        let (probe, mut events) = TelemetryProbe::channel(4096);
        let router = {
            let _guard = runtime.enter();
            spawn_router::<Flex>(server.clone(), AllowAll, probe)
        };
        let client = runtime.block_on(local_endpoint());
        let remote = {
            let _guard = runtime.enter();
            RemoteDevice::iroh(&client, server.addr(), 0)
        };
        remote.connect();
        let device = Device::new(remote);

        // The loader only uploads, so nothing else ever executes its stream.
        let (batches, received) = mpsc::sync_channel(2);
        let loader = {
            let device = device.clone();
            std::thread::spawn(move || {
                for step in 0..STEPS {
                    batches.send(Batch::new(step, &device)).unwrap();
                }
            })
        };
        for batch in received {
            consume(batch);
        }
        loader.join().unwrap();
        device.sync().unwrap();

        let mut seen = Vec::new();
        assert!(matches!(
            events.drain_into(&mut seen),
            DrainStatus::Open { lagged: 0 }
        ));
        let mut uploads = HashSet::new();
        let mut dropped = HashSet::new();
        for event in &seen {
            match event.as_ref() {
                TelemetryEvent::Op {
                    kind: OpClass::Init,
                    outputs,
                    ..
                } => uploads.extend(outputs.iter().map(|output| output.id)),
                TelemetryEvent::TensorDropped { tensor, .. } => {
                    dropped.insert(*tensor);
                }
                _ => {}
            }
        }
        assert_eq!(uploads.len(), STEPS * UPLOADS_PER_BATCH);
        assert_eq!(uploads.intersection(&dropped).count(), uploads.len());

        runtime.block_on(router.shutdown()).unwrap();
    }

    #[test]
    fn are_freed_when_computed_on_another_thread() {
        assert_server_drops_every_upload(|batch| {
            let loss = batch.images.sum() + batch.targets.float().sum();
            let _: f32 = loss.into_scalar();
        });
    }

    #[test]
    fn are_freed_when_read_on_another_thread() {
        assert_server_drops_every_upload(|batch| {
            batch.images.into_data();
            batch.targets.into_data();
        });
    }
}

mod iroh_peer {
    use super::*;
    use burn_remote::{
        ConnectError, EndpointId, IrohPeer, IrohPeerBuilder, IrohRelays, RemoteSecret,
        server::{Channel, IrohChannelBuilder, RemoteServerBuilder, TokenAuthorizer},
    };
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};

    const TOKEN: &str = "fleet-token";

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_with_the_token_reaches_a_relay_free_server_by_address() {
        let port = free_udp_port();
        let peer = direct_peer(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        let device = Device::new(peer.connect(0).await.unwrap());
        let data = Tensor::<1>::from_floats([4.0], &device) * 2.0;
        assert_eq!(data.try_into_vec_as::<f32>().unwrap(), vec![8.0]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_connected_twice_yields_the_same_device() {
        let port = free_udp_port();
        let peer = direct_peer(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        let first = peer.connect(0).await.unwrap();
        assert_eq!(peer.clone().connect(0).await.unwrap(), first);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_with_the_wrong_token_is_refused() {
        let port = free_udp_port();
        let id = serve_with_token(port);
        let admitted = direct_peer(id, Ipv4Addr::LOCALHOST.into(), port, TOKEN);
        admitted.connect(0).await.unwrap();

        let refused = direct_peer(id, Ipv4Addr::LOCALHOST.into(), port, "wrong-token");
        let panic = tokio::spawn(async move { refused.connect(0).await })
            .await
            .unwrap_err()
            .into_panic();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(
            message.contains("disconnected during initialization"),
            "{message}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn peers_built_from_clones_of_one_builder_bind_their_own_endpoints() {
        let port = free_udp_port();
        let builder = IrohPeerBuilder::new(serve_with_token(port))
            .with_relays(IrohRelays::Disabled)
            .with_address(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port))
            .with_credential(TOKEN);

        let first = builder.clone().build().connect(0).await.unwrap();
        assert_ne!(builder.build().connect(0).await.unwrap(), first);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_dials_from_an_application_endpoint() {
        let port = free_udp_port();
        let server = serve_with_token(port);
        let endpoint = local_endpoint().await;
        let peer = IrohPeerBuilder::new(server)
            .with_relays(IrohRelays::Disabled)
            .with_address(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port))
            .with_credential(TOKEN)
            .with_endpoint(endpoint.clone())
            .build();

        let device = peer.connect(0).await.unwrap();
        assert_eq!(peer.clone().connect(0).await.unwrap(), device);
        assert!(endpoint.remote_info(server).await.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_reaches_a_server_over_ipv6() {
        if UdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).is_err() {
            return;
        }
        let port = free_udp_port();
        let peer = direct_peer(
            serve_with_token(port),
            Ipv6Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        peer.connect(0).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_whose_ipv6_port_is_taken_still_serves_ipv4() {
        let port = free_udp_port();
        let Ok(_taken) = UdpSocket::bind((Ipv6Addr::LOCALHOST, port)) else {
            return;
        };
        let peer = direct_peer(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        peer.connect(0).await.unwrap();
    }

    #[tokio::test]
    async fn a_peer_without_relays_or_an_address_is_not_dialed() {
        let peer = IrohPeerBuilder::new(RemoteSecret::random().id())
            .with_relays(IrohRelays::Disabled)
            .build();

        assert!(matches!(
            peer.connect(0).await,
            Err(ConnectError::NoAddress)
        ));
    }

    fn free_udp_port() -> u16 {
        UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    fn serve_with_token(port: u16) -> EndpointId {
        let channel = IrohChannelBuilder::new(RemoteSecret::random())
            .with_relays(IrohRelays::Disabled)
            .with_port(port)
            .with_authorizer(TokenAuthorizer::new(TOKEN).unwrap())
            .build();
        let id = channel.id();
        tokio::spawn(
            RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .channel(Channel::Iroh { channel })
                .start_async(),
        );
        id
    }

    fn direct_peer(id: EndpointId, ip: std::net::IpAddr, port: u16, token: &str) -> IrohPeer {
        IrohPeerBuilder::new(id)
            .with_relays(IrohRelays::Disabled)
            .with_address(SocketAddr::new(ip, port))
            .with_credential(token)
            .build()
    }
}
