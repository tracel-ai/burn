#![cfg(all(feature = "client", feature = "server", feature = "websocket"))]

//! Graph capture on a remote device: the server records the graph on its own device and replays
//! it there, and a server whose backend records none answers the capture with its backend's error,
//! so the client re-runs the closure instead.
//!
//! The server hosts `Device::default()`: Flex here, which records no graph, and a GPU when one is
//! linked (`--features burn-tensor/rocm`, `burn-tensor/cuda` or `burn-tensor/vulkan`), which does.
//!
//! In a binary of its own, as `burn-backend-tests`' graph capture tests are: CUDA's global capture
//! mode can restrict API calls from other threads during capture.

use burn_dispatch::DispatchDevice;
use burn_ir::CapturedGraphId;
use burn_remote::RemoteChannel;
use burn_router::{RouterGraph, get_client};
use burn_tensor::{
    Device, Tensor,
    remote::RemoteHost,
    server::{RemoteServer, WebSocketTransport},
};

/// The input of [`step`], and what it makes of it.
const INPUT: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const STEP_OF_INPUT: [f32; 4] = [3.0, 5.0, 7.0, 9.0];

fn step(input: &Tensor<1>) -> Tensor<1> {
    input.clone().mul_scalar(2.0).add_scalar(1.0)
}

/// Whether the backend of `device` records a hardware graph of [`step`] itself.
fn records_a_graph(device: &Device) -> bool {
    let input = Tensor::<1>::from_floats(INPUT, device);
    burn_tensor::capture(device, || step(&input)).is_hardware()
}

/// Replaying a remote capture reproduces the eager result, repeatedly, and the graph is the
/// server's exactly when the server's backend records one.
///
/// Safety of the `replay` calls: the closure owns a clone of `input`, which keeps every captured
/// buffer alive as long as the graph, and the replays and reads run in turn on this thread.
#[test]
fn a_remote_capture_replays_on_the_server_device() {
    let server_device = Device::default();
    let server_records = records_a_graph(&server_device);
    let (rt, host) = serve(server_device);

    let device = Device::remote_options(&host).init().unwrap();
    let input = Tensor::<1>::from_floats(INPUT, &device);
    let mut graph = burn_tensor::capture(&device, || step(&input));

    println!(
        "server records a graph: {server_records}, remote capture: {:?}",
        graph
            .capture_error()
            .map_or("hardware".into(), |err| err.to_string())
    );
    assert_eq!(
        graph.is_hardware(),
        server_records,
        "{:?}",
        graph.capture_error()
    );
    if let Some(error) = graph.capture_error() {
        // The server backend's own words, carried back, rather than the router client's default.
        assert!(!error.to_string().contains("router client"), "{error}");
    }
    for _ in 0..3 {
        let out = unsafe { graph.replay() }.clone();
        assert_eq!(out.try_into_vec_as::<f32>().unwrap(), STEP_OF_INPUT);
    }

    drop(graph);
    rt.shutdown_background();
}

/// A graph reads its inputs where the capture found them, so refreshing an input in place
/// between replays is how a caller feeds it: each replay computes from the refreshed values.
/// Anything on the server that keeps an extra reference to a captured tensor stops the refresh
/// from writing in place, and every replay then returns the captured result.
///
/// Safety of the `replay` calls: the closure borrows `input`, which outlives the graph, and the
/// refreshes, replays and reads run in turn on this thread.
#[test]
fn a_remote_replay_sees_inputs_refreshed_in_place() {
    let server_device = Device::default();
    if !records_a_graph(&server_device) {
        println!("the server's backend records no graph; a re-run closure sees refreshes anyway");
        return;
    }
    let (rt, host) = serve(server_device);
    let device = Device::remote_options(&host).init().unwrap();

    let input = std::cell::RefCell::new(Tensor::<1>::from_floats(INPUT, &device));
    let mut graph = burn_tensor::capture(&device, || input.borrow().clone().mul_scalar(2.0));
    assert!(graph.is_hardware(), "{:?}", graph.capture_error());

    for value in [10.0, 20.0, 30.0] {
        input
            .borrow_mut()
            .inplace(|tensor| tensor.mul_scalar(0.0).add_scalar(value));
        let out = unsafe { graph.replay() }.clone();
        assert_eq!(out.try_into_vec_as::<f32>().unwrap(), [2.0 * value; 4]);
    }

    drop(graph);
    rt.shutdown_background();
}

/// A replay is not waited on, so one the server cannot run is the error of the next read on its
/// stream, rather than that read returning what the step before it wrote.
#[test]
fn a_failed_remote_replay_is_the_error_of_the_next_read() {
    let (rt, host) = serve(Device::default());
    let device = Device::remote_options(&host).init().unwrap();
    let DispatchDevice::Remote(remote) = device.as_dispatch() else {
        panic!("a remote device dispatches to the remote backend")
    };
    let input = Tensor::<1>::from_floats(INPUT, &device);

    // A graph the server never captured, which it fails to replay.
    let graph = RouterGraph::new(
        CapturedGraphId(u64::MAX),
        get_client::<RemoteChannel>(remote),
    );
    // Safety: the server holds no such graph, so the replay touches no buffer.
    unsafe { graph.replay() }.expect("a replay is only sent");

    let error = step(&input).try_into_vec_as::<f32>().unwrap_err();
    assert!(
        error.to_string().contains("unknown captured graph"),
        "{error}"
    );
    assert_eq!(
        step(&input).try_into_vec_as::<f32>().unwrap(),
        STEP_OF_INPUT,
        "the failure is reported once"
    );

    drop(graph);
    rt.shutdown_background();
}

/// A runtime serving `device` over a WebSocket on a free local port, and the host to dial.
fn serve(device: Device) -> (tokio::runtime::Runtime, RemoteHost) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.spawn(RemoteServer::new([device]).serve_async(WebSocketTransport::from_listener(listener)));
    (rt, host)
}
