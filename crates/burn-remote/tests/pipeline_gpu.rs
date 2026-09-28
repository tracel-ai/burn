//! Pipeline stages on this machine's GPU, across two servers, the shape a split model runs in.
//!
//! Ignored by default: it needs a Vulkan GPU. Run it with
//! `cargo test -p burn-remote --features gpu-tests --test pipeline_gpu -- --ignored`.
#![cfg(all(
    feature = "client",
    feature = "server",
    feature = "iroh",
    feature = "gpu-tests"
))]

use burn_cubecl::{Cube, cubecl};
use burn_remote::{
    BURN_REMOTE_ALPN, RemoteDevice,
    server::{AllowAll, CustomOpRegistry, IrohRemoteProtocol},
    telemetry::TelemetryProbe,
};
use burn_tensor::{Device, Distribution, Tensor};
use cubecl::device::WgpuDeviceKind;
use iroh::{Endpoint, EndpointAddr, RelayMode, endpoint::presets, protocol::Router};
use std::{
    net::{IpAddr, SocketAddr, UdpSocket},
    sync::Arc,
    time::Duration,
};

async fn local_endpoint() -> Endpoint {
    Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Disabled)
        .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
        .bind_addr(SocketAddr::from(([0, 0, 0, 0], 0)))
        .unwrap()
        .bind()
        .await
        .unwrap()
}

/// This machine's address on its network. Connecting a UDP socket sends nothing; it only picks the
/// route, and with it the local address.
fn lan_ip() -> IpAddr {
    let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.connect("192.0.2.1:9").unwrap();
    socket.local_addr().unwrap().ip()
}

/// A peer named the way a client outside the machine names it: the machine's own address, and not the
/// loopback one its servers share.
fn peer_on_the_network(endpoint: &Endpoint) -> EndpointAddr {
    let port = endpoint
        .addr()
        .ip_addrs()
        .next()
        .expect("the endpoint is bound to an IP address")
        .port();
    EndpointAddr::new(endpoint.id()).with_ip_addr(SocketAddr::new(lan_ip(), port))
}

/// This machine's first Vulkan GPU, whichever kind it is.
fn gpu() -> cubecl::Device {
    cubecl::Device::vulkan(WgpuDeviceKind::DiscreteGpu(0))
        .or_else(|_| cubecl::Device::vulkan(WgpuDeviceKind::IntegratedGpu(0)))
        .expect("this machine has a Vulkan GPU")
}

/// A server hosting `device` twice, so two stages on it run as two sessions, the way two cards of one
/// machine do.
fn spawn_server(endpoint: Endpoint, device: cubecl::Device) -> Router {
    let protocol = IrohRemoteProtocol::<Cube>::new(
        endpoint.clone(),
        vec![device.clone(), device],
        Arc::new(AllowAll),
        TelemetryProbe::disabled(),
        CustomOpRegistry::default(),
    );
    Router::builder(endpoint)
        .accept(BURN_REMOTE_ALPN, protocol)
        .spawn()
}

/// Three stages over a server on `first` and a server on `second`, the third returning to the first
/// server, which then holds two sessions: one exposing a tensor to the other server, one downloading
/// from it.
///
/// The runtime and the routers live until the process ends: dropping them while the sessions still
/// hold GPU streams crashes inside the driver.
fn three_stages(first: cubecl::Device, second: cubecl::Device) -> Vec<Device> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let guard = runtime.enter();

    let (near, far, client) = runtime.block_on(async {
        (
            local_endpoint().await,
            local_endpoint().await,
            local_endpoint().await,
        )
    });
    let near_addr = peer_on_the_network(&near);
    let far_addr = peer_on_the_network(&far);
    let routers = [spawn_server(near, first), spawn_server(far, second)];

    // One client endpoint for every stage, the way a runner holds one.
    let stages = [
        RemoteDevice::iroh(&client, near_addr.clone(), 0),
        RemoteDevice::iroh(&client, far_addr, 0),
        RemoteDevice::iroh(&client, near_addr, 1),
    ];
    for stage in &stages {
        stage.connect();
    }

    drop(guard);
    core::mem::forget((routers, runtime));
    stages.into_iter().map(Device::new).collect()
}

/// Work on every stage, and nothing read until the end of a pass, so transfers overlap the way they
/// do in a pipeline.
fn passes(stages: Vec<Device>, hidden: usize) -> impl FnOnce() + Send + 'static {
    move || {
        for _ in 0..5 {
            let mut tensor =
                Tensor::<2>::random([64, hidden], Distribution::Default, &stages[0]).tanh();
            for stage in &stages[1..] {
                tensor = tensor.to_device(stage);
                tensor = tensor.clone().matmul(tensor.transpose()).tanh();
            }
            let _ = tensor.into_data();
        }
    }
}

/// Fails the test if `body` has not finished by `timeout`: a stalled transfer parks a worker deep in
/// the backend, where it cannot be killed, so the process exit carries it away.
fn finishes_within(timeout: Duration, body: impl FnOnce() + Send + 'static) {
    let (done, waiting) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        body();
        let _ = done.send(());
    });
    waiting
        .recv_timeout(timeout)
        .unwrap_or_else(|_| panic!("the stages did not finish within {timeout:?}"));
}

#[test]
#[ignore = "needs a Vulkan GPU"]
fn stages_return_to_a_server_they_already_ran_on() {
    let stages = three_stages(gpu(), gpu());
    finishes_within(Duration::from_secs(120), passes(stages, 1024));
}

/// The middle stage runs on another runtime, so each hop crosses runtimes the way a hop between a
/// CUDA server and a Vulkan one does.
#[test]
#[ignore = "needs a Vulkan GPU"]
fn stages_cross_runtimes_and_return_to_a_server() {
    let cpu = cubecl::Device::cpu().expect("the cpu runtime is compiled in");
    let stages = three_stages(gpu(), cpu);
    finishes_within(Duration::from_secs(120), passes(stages, 256));
}
