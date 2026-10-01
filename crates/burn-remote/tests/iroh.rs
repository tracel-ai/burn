#![cfg(all(feature = "client", feature = "server", feature = "iroh"))]

use burn_flex::Flex;
use burn_ir::BackendIr;
use burn_remote::{
    BURN_REMOTE_ALPN,
    server::{AllowAll, IrohRemoteProtocol},
    telemetry::{TelemetryEvent, TelemetryProbe},
};
use burn_tensor::{
    DType, Device, Int, Tensor, TensorData,
    remote::{ConnectError, IrohHost, RemoteHost},
};
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

/// `server`, dialed from the test's own `client` endpoint.
fn host_dialed_from(client: &Endpoint, server: impl Into<EndpointAddr>) -> RemoteHost {
    RemoteHost::iroh(IrohHost::new(server).with_endpoint(client.clone()))
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

    let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
        .init_async()
        .await
        .unwrap();

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

    let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
        .init_async()
        .await
        .unwrap();
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

    let device = Device::remote_options(&host_dialed_from(&client, server.id()))
        .init_async()
        .await
        .unwrap();

    let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
    assert_eq!(output.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);

    router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_device_dialed_with_no_address_and_no_lookup_connects_once_given_one() {
    let server = local_endpoint().await;
    let router = spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled());
    let client = local_endpoint().await;

    let result = Device::remote_options(&host_dialed_from(&client, server.id()))
        .init_async()
        .await;
    assert!(matches!(result, Err(ConnectError::NoAddress)), "{result:?}");

    let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
        .init_async()
        .await
        .unwrap();
    let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
    assert_eq!(output.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);

    router.shutdown().await.unwrap();
}

#[test]
fn a_device_retried_after_its_first_runtime_shut_down_runs_on_the_new_one() {
    within_hang_limit(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(local_endpoint());
        let client = runtime.block_on(local_endpoint());
        let router = {
            let _guard = runtime.enter();
            spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled())
        };

        let first = tokio::runtime::Runtime::new().unwrap();
        let result = {
            let _guard = first.enter();
            Device::remote_options(&host_dialed_from(&client, server.id())).init()
        };
        assert!(matches!(result, Err(ConnectError::NoAddress)), "{result:?}");
        drop(first);

        let second = tokio::runtime::Runtime::new().unwrap();
        let device = {
            let _guard = second.enter();
            Device::remote_options(&host_dialed_from(&client, server.addr()))
                .init()
                .unwrap()
        };
        let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
        assert_eq!(output.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);

        runtime.block_on(router.shutdown()).unwrap();
    });
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

    let source = Device::remote_options(&host_dialed_from(&client, source_server.addr()))
        .init_async()
        .await
        .unwrap();
    let target = Device::remote_options(&host_dialed_from(&client, target_server.addr()))
        .init_async()
        .await
        .unwrap();

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
/// runtime in the calling code.
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

    let device = Device::remote_options(&host_dialed_from(&client_endpoint, server_addr))
        .init()
        .unwrap();

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
        let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
            .init()
            .unwrap();

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

            let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
                .init_async()
                .await
                .unwrap();
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

#[test]
fn tensors_dropped_on_another_thread_still_feed_their_queued_reader() {
    within_hang_limit(|| {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(local_endpoint());
        let client = runtime.block_on(local_endpoint());
        let router = {
            let _guard = runtime.enter();
            spawn_router::<Flex>(server.clone(), AllowAll, TelemetryProbe::disabled())
        };
        let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
            .init()
            .unwrap();

        let computed = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) + 1.0;
        // A free before the producer runs is a no-op, so only `computed` can catch an early free.
        device.sync().unwrap();
        let pending = computed.clone() * 2.0;
        let reader = pending.clone() + computed.clone();
        thread::spawn(move || drop((computed, pending)))
            .join()
            .unwrap();

        assert_eq!(
            reader.try_into_vec_as::<f32>().unwrap(),
            vec![6.0, 9.0, 12.0]
        );

        runtime.block_on(router.shutdown()).unwrap();
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
    let host = host_dialed_from(&client, server.addr()).with_credential(b"fleet-ticket".to_vec());
    let device = Device::remote_options(&host).init_async().await.unwrap();
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
    let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
        .init_async()
        .await
        .unwrap();

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
        let device = Device::remote_options(&host_dialed_from(&client, server.addr()))
            .init()
            .unwrap();

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
        EndpointId, IrohRelays, RemoteSecret,
        server::{
            AuthorizationRequest, Channel, IrohChannelBuilder, RemoteServerBuilder, TokenAuthorizer,
        },
    };
    use iroh_relay::server::{RelayConfig, Server, ServerConfig};
    use std::{
        net::{Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    const TOKEN: &str = "fleet-token";

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_reaches_a_server_through_a_private_relay() {
        let relay = private_relay().await;
        let relays = IrohRelays::Private {
            url: format!("http://{}", relay.http_addr().unwrap())
                .parse()
                .unwrap(),
        };
        let id = serve(IrohChannelBuilder::new(RemoteSecret::random()).with_relays(relays.clone()));
        let host = RemoteHost::iroh(IrohHost::new(id).with_relays(relays)).with_credential(TOKEN);

        Device::remote_options(&host).init_async().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_with_the_token_reaches_a_relay_free_server_by_address() {
        let port = free_udp_port();
        let host = direct_host(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        let device = Device::remote_options(&host).init_async().await.unwrap();
        let data = Tensor::<1>::from_floats([4.0], &device) * 2.0;
        assert_eq!(data.try_into_vec_as::<f32>().unwrap(), vec![8.0]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_connected_twice_yields_the_same_device() {
        let port = free_udp_port();
        let host = direct_host(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        let first = Device::remote_options(&host).init_async().await.unwrap();
        assert_eq!(
            Device::remote_options(&host).init_async().await.unwrap(),
            first
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_with_the_wrong_token_is_refused() {
        let port = free_udp_port();
        let id = serve_with_token(port);
        let admitted = direct_host(id, Ipv4Addr::LOCALHOST.into(), port, TOKEN);
        Device::remote_options(&admitted)
            .init_async()
            .await
            .unwrap();

        let refused = direct_host(id, Ipv4Addr::LOCALHOST.into(), port, "wrong-token");
        let result = Device::remote_options(&refused).init_async().await;
        assert!(
            matches!(result, Err(ConnectError::Unauthorized)),
            "{result:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_asking_for_a_device_the_server_lacks_is_told_how_many_it_hosts() {
        let port = free_udp_port();
        let host = direct_host(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        let result = Device::remote_options(&host)
            .device_index(1)
            .init_async()
            .await;
        assert!(
            matches!(
                result,
                Err(ConnectError::NoSuchDevice {
                    device_count: 1,
                    ..
                })
            ),
            "{result:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_refused_once_connects_on_its_next_try() {
        let port = free_udp_port();
        let open = Arc::new(AtomicBool::new(false));
        let admits = open.clone();
        let id = start(
            IrohChannelBuilder::new(RemoteSecret::random())
                .with_relays(IrohRelays::Disabled)
                .with_port(port)
                .with_authorizer(move |_: AuthorizationRequest<'_>| {
                    if admits.load(Ordering::Relaxed) {
                        Ok(())
                    } else {
                        Err("not yet".to_string())
                    }
                }),
        );
        let host = direct_host(id, Ipv4Addr::LOCALHOST.into(), port, TOKEN);

        let result = Device::remote_options(&host).init_async().await;
        assert!(
            matches!(result, Err(ConnectError::Unauthorized)),
            "{result:?}"
        );
        open.store(true, Ordering::Relaxed);
        Device::remote_options(&host).init_async().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_whose_server_is_not_at_the_address_cannot_reach_it() {
        let port = free_udp_port();
        serve_with_token(port);
        let elsewhere = RemoteSecret::random().id();
        let host = direct_host(elsewhere, Ipv4Addr::LOCALHOST.into(), port, TOKEN);

        let result = Device::remote_options(&host).init_async().await;
        assert!(
            matches!(&result, Err(ConnectError::Unreachable { .. })),
            "{result:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_dialed_from_two_application_endpoints_is_two_devices() {
        let port = free_udp_port();
        let server = serve_with_token(port);
        let dialed_from = |endpoint: Endpoint| {
            RemoteHost::iroh(
                IrohHost::new(server)
                    .with_address(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port))
                    .with_endpoint(endpoint),
            )
            .with_credential(TOKEN)
        };

        let first = dialed_from(local_endpoint().await);
        let second = dialed_from(local_endpoint().await);
        assert_ne!(
            Device::remote_options(&first).init_async().await.unwrap(),
            Device::remote_options(&second).init_async().await.unwrap()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_dials_from_an_application_endpoint() {
        let port = free_udp_port();
        let server = serve_with_token(port);
        let endpoint = local_endpoint().await;
        let host = RemoteHost::iroh(
            IrohHost::new(server)
                .with_address(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port))
                .with_endpoint(endpoint.clone()),
        )
        .with_credential(TOKEN);

        let device = Device::remote_options(&host).init_async().await.unwrap();
        assert_eq!(
            Device::remote_options(&host).init_async().await.unwrap(),
            device
        );
        assert!(endpoint.remote_info(server).await.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_peer_reaches_a_server_over_ipv6() {
        if UdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).is_err() {
            return;
        }
        let port = free_udp_port();
        let host = direct_host(
            serve_with_token(port),
            Ipv6Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        Device::remote_options(&host).init_async().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_server_whose_ipv6_port_is_taken_still_serves_ipv4() {
        let port = free_udp_port();
        let Ok(_taken) = UdpSocket::bind((Ipv6Addr::LOCALHOST, port)) else {
            return;
        };
        let host = direct_host(
            serve_with_token(port),
            Ipv4Addr::LOCALHOST.into(),
            port,
            TOKEN,
        );

        Device::remote_options(&host).init_async().await.unwrap();
    }

    #[tokio::test]
    async fn a_peer_without_relays_or_an_address_is_not_dialed() {
        let host = RemoteHost::iroh(
            IrohHost::new(RemoteSecret::random().id()).with_relays(IrohRelays::Disabled),
        );

        assert!(matches!(
            Device::remote_options(&host).init_async().await,
            Err(ConnectError::NoAddress)
        ));
    }

    /// A port free on IPv4, and on IPv6 where the host has it.
    fn free_udp_port() -> u16 {
        loop {
            let port = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
                .unwrap()
                .local_addr()
                .unwrap()
                .port();
            let ipv6 = UdpSocket::bind((Ipv6Addr::UNSPECIFIED, port));
            if ipv6.is_ok() || UdpSocket::bind((Ipv6Addr::UNSPECIFIED, 0)).is_err() {
                return port;
            }
        }
    }

    fn serve_with_token(port: u16) -> EndpointId {
        serve(
            IrohChannelBuilder::new(RemoteSecret::random())
                .with_relays(IrohRelays::Disabled)
                .with_port(port),
        )
    }

    fn serve(channel: IrohChannelBuilder) -> EndpointId {
        start(channel.with_authorizer(TokenAuthorizer::new(TOKEN).unwrap()))
    }

    fn start(channel: IrohChannelBuilder) -> EndpointId {
        let channel = channel.build();
        let id = channel.id();
        tokio::spawn(
            RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .channel(Channel::Iroh { channel })
                .start_async(),
        );
        id
    }

    /// A relay on a free local port, over plain HTTP so no certificate is needed.
    async fn private_relay() -> Server {
        let mut config = ServerConfig::default();
        config.relay = Some(RelayConfig::new((Ipv4Addr::LOCALHOST, 0)));
        Server::spawn(config).await.unwrap()
    }

    fn direct_host(id: EndpointId, ip: std::net::IpAddr, port: u16, token: &str) -> RemoteHost {
        RemoteHost::iroh(
            IrohHost::new(id)
                .with_relays(IrohRelays::Disabled)
                .with_address(SocketAddr::new(ip, port)),
        )
        .with_credential(token)
    }
}
