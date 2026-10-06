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
    Bool, DType, Device, DeviceType, Distribution, Int, Tensor, TensorData, Tolerance, Transaction,
    quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
    remote::RemoteHost,
    server::RemoteServer,
};

const TOKEN: &str = "fleet-token";

/// A 4 MiB tensor, which a session carries in several frames.
const MANY_FRAMES_LONG: usize = 1024 * 1024;

/// Far beyond what a bounded step here takes when it works, so only a hang reaches it.
const HANG_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Run `body` on a worker thread and fail the test if it does not finish within `timeout`.
///
/// A hung worker cannot be killed, so the test thread panics and the process exit takes it away.
fn with_deadlock_watchdog<T: Send + 'static>(
    timeout: std::time::Duration,
    body: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let value = body();
        let _ = tx.send(());
        value
    });
    match rx.recv_timeout(timeout) {
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("Deadlock: still blocked after {timeout:?}")
        }
        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => handle
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
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

/// Bind `address` again once a stopped server has freed it. Another test's socket on an
/// OS-picked port can hold it for a moment.
fn rebind(address: std::net::SocketAddr) -> std::net::TcpListener {
    for _ in 0..50 {
        match std::net::TcpListener::bind(address) {
            Ok(listener) => return listener,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(100)),
        }
    }
    panic!("{address} stayed taken after its server stopped");
}

/// A server restarted on its port: the host, and the device connected before the restart, seen
/// to have ended by a failed read.
fn restarted_server(rt: &tokio::runtime::Runtime) -> (RemoteHost, Device) {
    let serve = |listener| {
        rt.spawn(
            BackendServer::<Flex>::new(vec![Default::default()])
                .serve_async(WebSocketTransport::from_listener(listener)),
        )
    };
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let host = host_of(&listener);
    let first = serve(listener);

    let old = Device::remote_options(&host).init().unwrap();
    let doubled = Tensor::<1>::from_floats([1.0], &old) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![2.0]);

    first.abort();
    assert!(rt.block_on(first).unwrap_err().is_cancelled());
    // A read fails only once the client has seen the session end, which a reconnect relies on.
    let stale = old.clone();
    with_deadlock_watchdog(HANG_LIMIT, move || {
        let read = (Tensor::<1>::from_floats([1.0], &stale) * 2.0).try_into_data();
        assert!(read.is_err(), "a session outlived its server: {read:?}");
    });

    serve(rebind(address));
    (host, old)
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            panic
                .downcast_ref::<&str>()
                .map(|message| message.to_string())
        })
        .unwrap_or_default()
}

fn host_of(listener: &std::net::TcpListener) -> RemoteHost {
    RemoteHost::websocket(&format!("ws://{}", listener.local_addr().unwrap()))
}

/// The error `serving` stops with, which it must do within the hang limit.
fn serve_error(serving: impl Future<Output = Result<(), ServeError>>) -> ServeError {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async { tokio::time::timeout(HANG_LIMIT, serving).await })
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

/// A server reads at most 64 KiB from a client before admitting it, so a larger credential can
/// never pass, and is refused before a session is opened for it.
#[test]
fn a_credential_too_large_for_the_handshake_is_refused_before_connecting() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));

    let refused = Device::remote_options(&host.with_credential(vec![b'x'; 64 * 1024 + 1])).init();

    assert!(
        matches!(refused, Err(ConnectError::InvalidConfiguration { .. })),
        "{refused:?}"
    );
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

#[test]
fn a_transaction_returns_each_tensor_in_the_order_it_was_registered() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();

    let floats = Tensor::<1>::from_floats([1.0, 2.0], &device);
    let [first, ints, bools, second] = Transaction::default()
        .register(floats.clone())
        .register(Tensor::<1, Int>::from_ints([3, 4], &device))
        .register(Tensor::<1, Bool>::from_bool([true, false], &device))
        .register(floats * 10.0)
        .execute()
        .try_into()
        .unwrap();

    assert_eq!(first.iter::<f32>().collect::<Vec<_>>(), [1.0, 2.0]);
    assert_eq!(ints.iter::<i64>().collect::<Vec<_>>(), [3, 4]);
    assert_eq!(bools.iter::<bool>().collect::<Vec<_>>(), [true, false]);
    assert_eq!(second.iter::<f32>().collect::<Vec<_>>(), [10.0, 20.0]);

    rt.shutdown_background();
}

#[test]
fn a_transaction_reads_a_tensor_registered_twice() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();

    // The first read borrows the tensor and the second takes it, so their order matters.
    let tensor = Tensor::<1>::from_floats([1.0, 2.0], &device);
    let [first, second] = Transaction::default()
        .register(tensor.clone())
        .register(tensor)
        .execute()
        .try_into()
        .unwrap();

    assert_eq!(first.iter::<f32>().collect::<Vec<_>>(), [1.0, 2.0]);
    assert_eq!(second.iter::<f32>().collect::<Vec<_>>(), [1.0, 2.0]);

    rt.shutdown_background();
}

#[test]
fn a_transaction_reads_tensors_from_two_servers_in_order() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host_1 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let host_2 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device_1 = Device::remote_options(&host_1).init().unwrap();
    let device_2 = Device::remote_options(&host_2).init().unwrap();

    let [first, second, third] = Transaction::default()
        .register(Tensor::<1>::from_floats([1.0], &device_1))
        .register(Tensor::<1>::from_floats([2.0], &device_2))
        .register(Tensor::<1>::from_floats([3.0], &device_1))
        .execute()
        .try_into()
        .unwrap();

    assert_eq!(first.iter::<f32>().collect::<Vec<_>>(), [1.0]);
    assert_eq!(second.iter::<f32>().collect::<Vec<_>>(), [2.0]);
    assert_eq!(third.iter::<f32>().collect::<Vec<_>>(), [3.0]);

    rt.shutdown_background();
}

fn int8_scheme(device: &Device) -> QuantScheme {
    device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
}

fn quantized_data(device: &Device, scale: f32) -> TensorData {
    TensorData::quantized(
        vec![-127i8, -71, 0, 35],
        [4],
        int8_scheme(device),
        &[scale],
        None,
    )
}

#[test]
fn a_tensor_quantizes_and_dequantizes_on_the_server() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();

    let floats = Tensor::<1>::from_floats([5.0, 0.0, 4.0, -12.7], &device);
    let quantized = floats.clone().quantize_dynamic(&int8_scheme(&device));

    let expected = TensorData::quantized(
        vec![50i8, 0, 40, -127],
        [4],
        int8_scheme(&device).with_store(QuantStore::Native),
        &[0.1],
        None,
    );
    quantized.to_data().assert_eq(&expected, false);
    quantized
        .dequantize()
        .into_data()
        .assert_approx_eq::<f32>(&floats.into_data(), Tolerance::absolute(1e-1));

    rt.shutdown_background();
}

#[test]
fn quantized_data_reads_back_as_it_was_uploaded() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();

    let data = quantized_data(&device, 0.014_173_228);
    let tensor = Tensor::<1>::from_data(data.clone(), &device);

    tensor.into_data().assert_eq(&data, true);

    rt.shutdown_background();
}

#[test]
fn a_quantized_tensor_stays_quantized_through_a_transpose() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();

    let floats = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let transposed = floats
        .clone()
        .quantize_dynamic(&int8_scheme(&device))
        .transpose();

    let data = transposed.to_data();
    assert!(
        matches!(data.dtype(), DType::QFloat(_)),
        "{:?}",
        data.dtype()
    );
    transposed
        .dequantize()
        .into_data()
        .assert_approx_eq::<f32>(&floats.transpose().into_data(), Tolerance::absolute(1e-1));

    rt.shutdown_background();
}

#[test]
fn quantized_operands_multiply_on_the_server() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();
    let scheme = int8_scheme(&device);

    let lhs = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device).quantize_dynamic(&scheme);
    let rhs = Tensor::<2>::from_floats([[2.0, 0.0], [1.0, 2.0]], &device).quantize_dynamic(&scheme);

    lhs.matmul(rhs).into_data().assert_approx_eq::<f32>(
        &TensorData::from([[4.0, 4.0], [10.0, 8.0]]),
        Tolerance::relative(2e-2),
    );

    rt.shutdown_background();
}

#[test]
fn a_quantized_tensor_moves_between_servers() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host_1 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let host_2 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device_1 = Device::remote_options(&host_1).init().unwrap();
    let device_2 = Device::remote_options(&host_2).init().unwrap();

    let data = quantized_data(&device_1, 0.014_173_228);
    let moved = Tensor::<1>::from_data(data.clone(), &device_1).to_device(&device_2);

    moved.into_data().assert_eq(&data, true);

    rt.shutdown_background();
}

#[test]
fn a_quantized_tensor_moves_between_devices_of_one_server() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(
        &rt,
        BackendServer::<Flex>::new(vec![Default::default(), Default::default()]),
    );
    let device_0 = Device::remote_options(&host).init().unwrap();
    let device_1 = Device::remote_options(&host)
        .device_index(1)
        .init()
        .unwrap();

    let data = quantized_data(&device_0, 0.014_173_228);
    let moved = Tensor::<1>::from_data(data.clone(), &device_0).to_device(&device_1);

    moved.into_data().assert_eq(&data, true);

    rt.shutdown_background();
}

/// Every layout a quantized tensor crosses the wire in: one scale, block scales, values packed
/// eight to a word, and block scales under a per-tensor scale.
fn wire_schemes(device: &Device) -> [QuantScheme; 4] {
    let q8 = int8_scheme(device);
    [
        q8,
        q8.per_block([32], ScaleDtype::F32),
        q8.with_value(QuantValue::Q4S)
            .with_store(QuantStore::PackedU32(0))
            .per_block([32], ScaleDtype::F32),
        q8.per_block([2, 16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32),
    ]
}

/// Values whose magnitude changes along the tensor, so every block gets a scale of its own.
fn wire_floats() -> TensorData {
    let values: Vec<f32> = (0..64 * 64)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) * (1 + i / 512) as f32 * 0.01)
        .collect();
    TensorData::new(values, [64, 64])
}

/// Values, scales and scheme alike: the bytes a quantized tensor is sent as.
#[track_caller]
fn assert_same_bytes(actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.as_bytes(), expected.as_bytes());
}

#[test]
fn quantized_data_crosses_the_wire_byte_for_byte() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();
    let local = Device::flex();

    for scheme in wire_schemes(&device) {
        let quantized = Tensor::<2>::from_data(wire_floats(), &local).quantize_dynamic(&scheme);
        let expected = quantized.clone().into_data();

        let quantized_remotely = Tensor::<2>::from_data(wire_floats(), &device)
            .quantize_dynamic(&scheme)
            .into_data();
        assert_same_bytes(&quantized_remotely, &expected);

        let uploaded = Tensor::<2>::from_data(expected.clone(), &device);
        assert_same_bytes(&uploaded.clone().into_data(), &expected);
        uploaded
            .dequantize()
            .into_data()
            .assert_eq(&quantized.dequantize().into_data(), true);
    }

    rt.shutdown_background();
}

#[test]
fn quantized_data_moves_between_servers_byte_for_byte() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host_1 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let host_2 = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device_1 = Device::remote_options(&host_1).init().unwrap();
    let device_2 = Device::remote_options(&host_2).init().unwrap();
    let local = Device::flex();

    for scheme in wire_schemes(&device_1) {
        let expected = Tensor::<2>::from_data(wire_floats(), &local)
            .quantize_dynamic(&scheme)
            .into_data();

        let moved = Tensor::<2>::from_data(expected.clone(), &device_1).to_device(&device_2);

        assert_same_bytes(&moved.into_data(), &expected);
    }

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
    let (host, old) = restarted_server(&rt);

    let new = with_deadlock_watchdog(HANG_LIMIT, move || {
        Device::remote_options(&host).init().unwrap()
    });
    assert_ne!(new, old);
    let doubled = Tensor::<1>::from_floats([3.0], &new) * 2.0;
    assert_eq!(doubled.try_into_vec_as::<f32>().unwrap(), vec![6.0]);

    for (from, to) in [(&old, &new), (&new, &old)] {
        let moved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Tensor::<1>::from_floats([1.0], from).to_device(to)
        }))
        .expect_err("a tensor moved across a device whose session ended");
        let message = panic_message(moved);
        assert!(message.contains("session has ended"), "{message}");
    }

    rt.shutdown_background();
}

fn loopback_transport() -> WebSocketTransport {
    WebSocketTransport::from_listener(std::net::TcpListener::bind("127.0.0.1:0").unwrap())
}

#[test]
fn a_remote_server_with_no_devices_refuses_to_serve() {
    let error =
        serve_error(RemoteServer::new(Vec::<Device>::new()).serve_async(loopback_transport()));
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

    let error = serve_error(RemoteServer::new([remote]).serve_async(loopback_transport()));
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
        BackendServer::<Flex>::new(vec![Default::default()])
            .serve_async(WebSocketTransport::new(port)),
    );
    assert!(matches!(error, ServeError::Bind { .. }), "{error:?}");
}

#[test]
fn a_blocking_serve_on_a_taken_port_returns_a_bind_error() {
    let taken = std::net::TcpListener::bind("0.0.0.0:0").unwrap();
    let port = taken.local_addr().unwrap().port();

    let error = RemoteServer::new([Device::flex()])
        .serve(WebSocketTransport::new(port))
        .unwrap_err();
    assert!(matches!(error, ServeError::Bind { .. }), "{error:?}");
}

#[test]
fn a_server_with_no_devices_refuses_to_serve() {
    let error =
        serve_error(BackendServer::<Flex>::new(Vec::new()).serve_async(loopback_transport()));
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

#[test]
fn a_tensor_many_frames_long_is_uploaded_and_read_back() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .build()
        .unwrap();
    let host = serve(&rt, BackendServer::<Flex>::new(vec![Default::default()]));
    let device = Device::remote_options(&host).init().unwrap();
    let values: Vec<f32> = (0..MANY_FRAMES_LONG).map(|i| i as f32).collect();

    let tensor = Tensor::<1>::from_data(TensorData::new(values.clone(), [values.len()]), &device);
    let doubled = (tensor * 2.0).try_into_vec_as::<f32>().unwrap();

    let expected: Vec<f32> = values.iter().map(|value| value * 2.0).collect();
    assert_eq!(doubled, expected);
    rt.shutdown_background();
}
