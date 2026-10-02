#![cfg(all(feature = "client", feature = "server", feature = "websocket"))]

use burn_flex::Flex;
use burn_remote::{
    ConnectError,
    server::{
        AuthorizationRequest, BackendServer, ClientId, ServeError, TokenAuthorizer,
        WebSocketTransport,
    },
};
use burn_tensor::{
    Device, DeviceType, Distribution, Tensor, remote::RemoteHost, server::RemoteServer,
};

const TOKEN: &str = "fleet-token";

/// Far beyond what a bounded step here takes when it works, so only a hang reaches it.
const HANG_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Run `body` on a worker thread and fail the test if it does not finish within `timeout`.
///
/// A hung worker cannot be killed, so the test thread panics and the process exit takes it away.
fn with_deadlock_watchdog(timeout: std::time::Duration, body: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        body();
        let _ = tx.send(());
    });
    match rx.recv_timeout(timeout) {
        Ok(()) => {
            handle.join().expect("worker thread panicked");
        }
        Err(_) => {
            panic!("Deadlock: the remote multi-device workload did not finish within {timeout:?}")
        }
    }
}

/// Serve `server` over WebSocket on a port the OS picks, returning the host to dial.
///
/// The listener is bound before this returns, so a client can connect at once.
fn serve(rt: &tokio::runtime::Runtime, server: BackendServer<Flex>) -> RemoteHost {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = host_of(&listener);
    let serving = server.serve_async(WebSocketTransport::from_listener(listener));
    rt.spawn(async move { serving.await.unwrap() });
    host
}

fn host_of(listener: &std::net::TcpListener) -> RemoteHost {
    RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()))
}

fn serve_error(server: BackendServer<Flex>, transport: WebSocketTransport) -> ServeError {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async { tokio::time::timeout(HANG_LIMIT, server.serve_async(transport)).await })
        .expect("the server is still serving")
        .unwrap_err()
}

#[test]
fn a_device_the_server_does_not_host_is_an_error() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));

    let result = Device::remote_options(&host).device_index(1).init();

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
    rt.shutdown_background();
}

#[test]
fn only_a_websocket_client_with_the_token_is_admitted() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = serve(
        &rt,
        BackendServer::<Flex>::new(vec![Default::default()])
            .with_authorizer(TokenAuthorizer::new(TOKEN).unwrap()),
    );

    let refused = Device::remote_options(&host.clone().with_credential("wrong-token")).init();
    assert!(
        matches!(refused, Err(ConnectError::Unauthorized)),
        "{refused:?}"
    );
    Device::remote_options(&host.with_credential(TOKEN))
        .init()
        .unwrap();

    rt.shutdown_background();
}

#[test]
fn a_websocket_authorizer_sees_the_client_by_its_address() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let (seen, clients) = std::sync::mpsc::channel();
    let host = serve(
        &rt,
        BackendServer::<Flex>::new(vec![Default::default()]).with_authorizer(
            move |request: AuthorizationRequest<'_>| {
                seen.send(request.client).map_err(|err| err.to_string())
            },
        ),
    );

    Device::remote_options(&host).init().unwrap();

    let client = clients.try_recv().unwrap();
    assert!(
        matches!(client, ClientId::WebSocket(address) if address.ip().is_loopback()),
        "{client:?}"
    );
    rt.shutdown_background();
}

#[test]
fn a_server_that_never_starts_is_unreachable_once_the_retries_run_out() {
    // Bound but never listening: every dial is refused, and no other socket can take the port.
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let host = RemoteHost::websocket(&format!("ws://{}", socket.local_addr().unwrap()));

    with_deadlock_watchdog(std::time::Duration::from_secs(60), move || {
        let result = Device::remote_options(&host).init();
        assert!(
            matches!(result, Err(ConnectError::Unreachable { .. })),
            "{result:?}"
        );
    });
    drop(socket);
}

#[test]
fn a_dial_waits_for_a_websocket_server_that_starts_late() {
    // Past the first retries, well inside the retry window.
    const SERVER_LATE_BY: std::time::Duration = std::time::Duration::from_millis(700);
    const LISTEN_BACKLOG: u32 = 128;

    // Bound but not listening: dials are refused, and no other socket can take the port.
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let host = RemoteHost::websocket(&format!("ws://{}", socket.local_addr().unwrap()));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.spawn(async move {
        tokio::time::sleep(SERVER_LATE_BY).await;
        let listener = socket.listen(LISTEN_BACKLOG).unwrap().into_std().unwrap();
        BackendServer::<Flex>::new(vec![Default::default()])
            .serve_async(WebSocketTransport::from_listener(listener))
            .await
            .unwrap();
    });

    with_deadlock_watchdog(std::time::Duration::from_secs(30), move || {
        let device = Device::remote_options(&host).init().unwrap();
        let output = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
        assert_eq!(output.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);
    });

    rt.shutdown_background();
}

#[test]
fn test_to_device_over_websocket() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    let host_1 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let host_2 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));

    let device_1 = Device::remote_options(&host_1).init().unwrap();
    let device_2 = Device::remote_options(&host_2).init().unwrap();

    // Some random input on device 1.
    let input_shape = [1, 28, 28];
    let input = Tensor::<3>::random(input_shape, Distribution::Default, &device_1);
    let numbers_expected: Vec<f32> = input.to_data().try_into_vec().unwrap();

    // Move tensor to device 2.
    let input = input.to_device(&device_2);
    let numbers: Vec<f32> = input.to_data().try_into_vec().unwrap();
    assert_eq!(numbers, numbers_expected);

    // Move tensor back to device 1.
    let input = input.to_device(&device_1);
    let numbers: Vec<f32> = input.into_data().try_into_vec().unwrap();
    assert_eq!(numbers, numbers_expected);

    rt.shutdown_background();
}

/// The server's backend has no device clock, so it opens no profiling window and the client
/// measures between two syncs instead.
#[test]
fn test_profile_over_websocket() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));

    let device = Device::remote_options(&host).init().unwrap();
    let (sum, duration) = device
        .profile(|| {
            Tensor::<1>::ones([1024], &device)
                .sum()
                .into_scalar::<f32>()
        })
        .expect("a window the server cannot open is measured between syncs");

    assert_eq!(sum, 1024.0);
    let ticks = burn_std::future::block_on(duration.resolve())
        .expect("a system-time window always carries a measurement");
    assert!(ticks.duration() > std::time::Duration::ZERO);

    rt.shutdown_background();
}

/// A single server hosting multiple devices: two indices on the same address resolve to
/// two distinct sessions (distinct interpreters/runner threads). Moving a tensor between
/// them exercises the multi-device path within one host.
#[test]
fn test_multi_device_single_server() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    // One server, two devices.
    let host = serve(
        &rt,
        BackendServer::<Flex>::new(vec![Default::default(), Default::default()]),
    );

    let device_0 = Device::remote_options(&host).init().unwrap();
    let device_1 = Device::remote_options(&host)
        .device_index(1)
        .init()
        .unwrap();

    // Distinct indices on the same address must be distinct devices.
    assert_ne!(device_0, device_1);

    let input_shape = [1, 28, 28];
    let input = Tensor::<3>::random(input_shape, Distribution::Default, &device_0);
    let numbers_expected: Vec<f32> = input.to_data().try_into_vec().unwrap();

    // Move tensor to the second device on the same host and back.
    let input = input.to_device(&device_1);
    let numbers: Vec<f32> = input.to_data().try_into_vec().unwrap();
    assert_eq!(numbers, numbers_expected);

    let input = input.to_device(&device_0);
    let numbers: Vec<f32> = input.into_data().try_into_vec().unwrap();
    assert_eq!(numbers, numbers_expected);

    rt.shutdown_background();
}

/// Two threads, each pinned to a device, move tensors to the other device and back at once, so
/// two same-host transfers are in flight together and need distinct transfer ids.
#[test]
fn test_multi_device_concurrent_to_device_deadlock() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    let host = serve(
        &rt,
        BackendServer::<Flex>::new(vec![Default::default(), Default::default()]),
    );

    with_deadlock_watchdog(std::time::Duration::from_secs(30), move || {
        let device0 = Device::remote_options(&host).init().unwrap();
        let device1 = Device::remote_options(&host)
            .device_index(1)
            .init()
            .unwrap();

        let run = |home: Device, away: Device| {
            move || {
                for _ in 0..100 {
                    let t = Tensor::<2>::random([8, 8], Distribution::Default, &home);
                    // home -> away, op there, away -> home.
                    let t = t.to_device(&away);
                    let t = t * 2.0;
                    let t = t.to_device(&home);
                    let _ = t.sum().into_data();
                }
            }
        };

        let h0 = std::thread::spawn(run(device0.clone(), device1.clone()));
        let h1 = std::thread::spawn(run(device1, device0));
        h0.join().unwrap();
        h1.join().unwrap();
    });

    rt.shutdown_background();
}

/// `RemoteHost::devices` lists every device the server hosts, by connecting once and reading the
/// device count off the init handshake.
#[test]
fn test_enumerate_remote_devices() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    // One server hosting three devices.
    let host = serve(
        &rt,
        BackendServer::<Flex>::new(vec![
            Default::default(),
            Default::default(),
            Default::default(),
        ]),
    );

    let devices = Device::enumerate(DeviceType::Remote(host.clone())).into_vec();
    assert_eq!(host.devices().unwrap().into_vec(), devices);

    // The server reports its three devices, in index order.
    assert_eq!(devices.len(), 3);
    assert_eq!(devices[0], Device::remote_options(&host).init().unwrap());
    assert_eq!(
        devices[1],
        Device::remote_options(&host)
            .device_index(1)
            .init()
            .unwrap()
    );
    assert_eq!(
        devices[2],
        Device::remote_options(&host)
            .device_index(2)
            .init()
            .unwrap()
    );

    // Distinct indices are distinct devices.
    assert_ne!(devices[0], devices[1]);
    assert_ne!(devices[1], devices[2]);

    // The enumerated devices are usable: run an op on the last one.
    let input = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &devices[2]);
    let numbers: Vec<f32> = (input * 2.0).into_data().try_into_vec().unwrap();
    assert_eq!(numbers, vec![2.0, 4.0, 6.0, 8.0]);

    rt.shutdown_background();
}

/// Run `body` on a worker thread and report whether it finished within `timeout`.
///
/// Unlike [`with_deadlock_watchdog`], a panic inside `body` counts as finished: a failure that
/// surfaces is what these tests want, only a hang fails them.
fn finishes_within(timeout: std::time::Duration, body: impl FnOnce() + Send + 'static) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        let _ = tx.send(());
    });
    rx.recv_timeout(timeout).is_ok()
}

#[test]
fn test_server_down_does_not_hang_client() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));

    let device = Device::remote_options(&host).init().unwrap();

    // One successful round-trip so the sockets are actually up and the demux task is running.
    let input = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let warmup: Vec<f32> = (input * 2.0).into_data().try_into_vec().unwrap();
    assert_eq!(warmup, vec![2.0, 4.0, 6.0, 8.0]);

    // Kill the server: dropping its runtime closes the listener and both client sockets.
    rt.shutdown_timeout(std::time::Duration::from_millis(100));
    // Let the client's response-demux observe the closed stream and fail pending callers.
    std::thread::sleep(std::time::Duration::from_millis(500));

    // A read now has no server to answer it. It must error out (which `to_data` surfaces as a
    // panic), not hang.
    let finished = finishes_within(std::time::Duration::from_secs(10), move || {
        let t = Tensor::<2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
        let _ = (t * 2.0).to_data();
    });
    assert!(
        finished,
        "client hung waiting for a response after the server went down"
    );
}

#[test]
fn dropping_the_serving_future_ends_its_live_sessions() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = host_of(&listener);
    let server = rt.spawn(
        BackendServer::<Flex>::new(vec![Default::default()])
            .serve_async(WebSocketTransport::from_listener(listener)),
    );

    let device = Device::remote_options(&host).init().unwrap();
    let doubled = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);

    server.abort();
    assert!(rt.block_on(server).unwrap_err().is_cancelled());

    with_deadlock_watchdog(HANG_LIMIT, move || {
        let read = (Tensor::<1>::from_floats([3.0], &device) * 2.0).try_into_data();
        assert!(read.is_err(), "a session outlived its server: {read:?}");
    });
    rt.shutdown_background();
}

#[test]
fn a_device_whose_server_restarted_is_replaced_by_a_new_one() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let host = host_of(&listener);
    let first = rt.spawn(
        BackendServer::<Flex>::new(vec![Default::default()])
            .serve_async(WebSocketTransport::from_listener(listener)),
    );

    let old = Device::remote_options(&host).init().unwrap();
    let doubled = Tensor::<1>::from_floats([1.0], &old) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![2.0]);

    first.abort();
    assert!(rt.block_on(first).unwrap_err().is_cancelled());
    let stale = old.clone();
    with_deadlock_watchdog(HANG_LIMIT, move || {
        let read = (Tensor::<1>::from_floats([1.0], &stale) * 2.0).try_into_data();
        assert!(read.is_err(), "a session outlived its server: {read:?}");
    });

    let listener = std::net::TcpListener::bind(address).unwrap();
    rt.spawn(
        BackendServer::<Flex>::new(vec![Default::default()])
            .serve_async(WebSocketTransport::from_listener(listener)),
    );
    let new = Device::remote_options(&host).init().unwrap();
    assert_ne!(new, old);
    let doubled = Tensor::<1>::from_floats([3.0], &new) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![6.0]);

    let moved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Tensor::<1>::from_floats([1.0], &old).to_device(&new)
    }));
    assert!(moved.is_err(), "a tensor left a device whose session ended");

    rt.shutdown_background();
}

fn remote_server_error(server: RemoteServer) -> ServeError {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        tokio::time::timeout(
            HANG_LIMIT,
            server.serve_async(WebSocketTransport::from_listener(listener)),
        )
        .await
    })
    .expect("the server is still serving")
    .unwrap_err()
}

#[test]
fn a_remote_server_with_no_devices_refuses_to_serve() {
    let error = remote_server_error(RemoteServer::new(Vec::<Device>::new()));
    assert!(matches!(error, ServeError::NoDevices), "{error:?}");
}

#[test]
fn a_remote_device_cannot_be_served() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let remote = Device::remote_options(&host).init().unwrap();

    let error = remote_server_error(RemoteServer::new([remote]));
    assert!(
        matches!(error, ServeError::UnsupportedDevice { .. }),
        "{error:?}"
    );
    rt.shutdown_background();
}

#[test]
fn a_remote_server_serves_its_devices() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = host_of(&listener);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.spawn(
        RemoteServer::new([Device::flex()])
            .serve_async(WebSocketTransport::from_listener(listener)),
    );

    let device = Device::remote_options(&host).init().unwrap();
    let doubled = Tensor::<1>::from_floats([1.0, 2.0], &device) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![2.0, 4.0]);
    rt.shutdown_background();
}

/// Sends the process `SIGTERM`, which only the server under test is listening for.
#[cfg(unix)]
#[test]
fn a_blocking_serve_returns_once_the_process_is_told_to_stop() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = host_of(&listener);
    let (stopped, result) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let served =
            RemoteServer::new([Device::flex()]).serve(WebSocketTransport::from_listener(listener));
        let _ = stopped.send(served);
    });

    // A session that opens means the server is up, and its signal handlers are installed.
    let device = Device::remote_options(&host).init().unwrap();
    let doubled = Tensor::<1>::from_floats([1.0], &device) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![2.0]);

    let status = std::process::Command::new("kill")
        .args(["-TERM", &std::process::id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());

    let served = result
        .recv_timeout(HANG_LIMIT)
        .expect("the server kept serving after SIGTERM");
    assert!(served.is_ok(), "{served:?}");
}

#[test]
fn serving_on_a_taken_port_is_a_bind_error() {
    // The server binds every interface, and only that same address is refused on every OS.
    let taken = std::net::TcpListener::bind("0.0.0.0:0").unwrap();
    let port = taken.local_addr().unwrap().port();

    let error = serve_error(
        BackendServer::<Flex>::new(vec![Default::default()]),
        WebSocketTransport::new(port),
    );
    assert!(matches!(error, ServeError::Bind { .. }), "{error:?}");
}

#[test]
fn a_server_with_no_devices_refuses_to_serve() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();

    let error = serve_error(
        BackendServer::<Flex>::new(Vec::new()),
        WebSocketTransport::from_listener(listener),
    );
    assert!(matches!(error, ServeError::NoDevices), "{error:?}");
}

/// The tensor crosses backends as `TensorData`: local to remote, an op there, then back.
#[test]
fn test_to_device_local_to_remote() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();

    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));

    let local = Device::flex();
    let remote = Device::remote_options(&host).init().unwrap();

    // Create on local, move to remote.
    let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &local);
    let on_remote = input.clone().to_device(&remote);

    // Run an op while on the remote.
    let doubled = on_remote * 2.0;

    // Move back to local and verify.
    let back = doubled.to_device(&local);
    let numbers: Vec<f32> = back.into_data().try_into_vec().unwrap();
    assert_eq!(numbers, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    rt.shutdown_background();
}
