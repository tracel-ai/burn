#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "server")]
pub mod server;

pub(crate) mod shared;

pub use burn_communication::Protocol;
pub use burn_ir as ir;
pub use burn_router::RouterClient;
pub use shared::RemoteProtocol;

/// Network-traffic savings metric for op-graph caching, shared by the client device service and the
/// server session worker.
#[cfg(any(feature = "client", feature = "server"))]
pub(crate) mod metrics;

#[cfg(feature = "client")]
mod __client {
    use super::*;

    use crate::{client::RemoteChannel, shared::RemoteProtocol};
    use burn_communication::Protocol;
    use burn_router::BackendRouter;

    /// The remote backend allows you to run computation on a remote device.
    ///
    /// Make sure there is a running server before trying to connect to it.
    /// The recommended way to start one is via `burn::server::start` (requires
    /// the `server` feature on `burn`):
    ///
    /// ```rust, ignore
    /// use burn::{Device, server::{start, Channel}};
    ///
    /// start(Device::default(), Channel::WebSocket { port: 3000 });
    /// ```
    ///
    /// For backends that aren't part of `DispatchDevice` but implement
    /// `BackendIr`, build a [`server::RemoteServerBuilder`] directly with the
    /// concrete backend type parameter — that is also how custom operations
    /// (backend extensions) are hosted, via
    /// [`custom_op`](server::RemoteServerBuilder::custom_op).
    #[cfg(not(feature = "fusion"))]
    pub type RemoteBackend = BackendRouter<RemoteChannel<<RemoteProtocol as Protocol>::Client>>;

    /// With the `fusion` feature enabled, the remote backend is wrapped in
    /// [`Fusion`](burn_fusion::Fusion) — exactly like the CubeCL backends — so recurring groups of
    /// operations are cached on the server and invoked by id, sending a repeated computation
    /// (e.g. a model block per step) over the network once instead of every step.
    #[cfg(feature = "fusion")]
    pub type RemoteBackend =
        burn_fusion::Fusion<BackendRouter<RemoteChannel<<RemoteProtocol as Protocol>::Client>>>;

    pub use client::{CustomOpClient, RemoteDevice};
}
#[cfg(feature = "client")]
pub use __client::*;

#[cfg(all(test, feature = "client", feature = "server"))]
mod tests {
    use burn_flex::Flex;
    use burn_tensor::{Device, DeviceType, Distribution, Tensor};

    /// Run `body` on a worker thread and fail the test if it doesn't finish within `timeout`.
    ///
    /// A deadlock in the remote backend manifests as a hung worker, so without a watchdog the
    /// test would block the whole suite forever. We can't forcibly kill the hung thread (it's
    /// parked on a blocking recv deep in the backend), so on timeout we panic from the test
    /// thread and let the process exit carry the stuck worker away.
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
            Err(_) => panic!(
                "Deadlock: the remote multi-device workload did not finish within {timeout:?}"
            ),
        }
    }

    #[test]
    pub fn test_to_device_over_websocket() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3000)
                .start_async(),
        );
        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3010)
                .start_async(),
        );

        // Give the servers a moment to bind before clients try to connect.
        std::thread::sleep(std::time::Duration::from_millis(500));

        let device_1 = Device::remote("ws://localhost:3000", 0);
        let device_2 = Device::remote("ws://localhost:3010", 0);

        // Some random input on device 1.
        let input_shape = [1, 28, 28];
        let input = Tensor::<3>::random(input_shape, Distribution::Default, &device_1);
        let numbers_expected: Vec<f32> = input.to_data().to_vec().unwrap();

        // Move tensor to device 2.
        let input = input.to_device(&device_2);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        // Move tensor back to device 1.
        let input = input.to_device(&device_1);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        rt.shutdown_background();
    }

    /// End-to-end backend extension over the wire: the client ships a custom op as
    /// `OperationIr::Custom`, and the server executes it through a handler registered on the
    /// builder. Mirrors how a backend extension hosts its ops — the user hand-writes the client
    /// side (here, building the `CustomOpIr`) and registers the server handler.
    ///
    /// Only runs without `fusion`, since it drives the router client (`RemoteBackend`) directly.
    #[test]
    #[cfg(not(feature = "fusion"))]
    pub fn test_custom_op_over_websocket() {
        use crate::{RemoteBackend, RemoteDevice};
        use burn_backend::{Scalar, TensorData, TensorMetadata, ops::FloatTensorOps};
        use burn_ir::{CustomOpIr, OperationIr, ScalarIr, TensorIr};
        use burn_router::RouterClient;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        // Host a "scale" custom op: multiply the input float tensor by a scalar argument.
        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3200)
                .custom_op("scale", |handles, ir, _device| {
                    let input = handles.get_float_tensor::<Flex>(&ir.inputs[0]);
                    let factor: Scalar = ir.scalars[0].into();
                    let output = Flex::float_mul_scalar(input, factor);
                    handles.register_float_tensor::<Flex>(&ir.outputs[0].id, output);
                })
                .start_async(),
        );

        // Give the server a moment to bind before the client connects.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Drive the remote backend directly (no autodiff/dispatch glue). A real backend extension
        // would wrap this in a hand-written `impl MyExt for RemoteBackend`.
        let device = RemoteDevice::new("ws://localhost:3200", 0);
        let input = <RemoteBackend as FloatTensorOps<RemoteBackend>>::float_from_data(
            TensorData::from([2.0f32, 4.0, 6.0]),
            &device,
        );

        // Client side: build the custom op (input tensor + the scale factor as a scalar) and ship
        // it through the remote client as `OperationIr::Custom`.
        let client = input.client.clone();
        let shape = input.shape();
        let dtype = input.dtype();
        let out_ir = TensorIr::uninit(client.create_empty_handle(), shape, dtype);
        let desc = CustomOpIr::with_scalars(
            "scale",
            &[input.into_ir()],
            &[out_ir],
            vec![ScalarIr::Float(3.0)],
        );
        let out = client.register(OperationIr::Custom(desc)).remove(0);

        let data = rt
            .block_on(<RemoteBackend as FloatTensorOps<RemoteBackend>>::float_into_data(out))
            .unwrap();
        let values: Vec<f32> = data.to_vec().unwrap();
        assert_eq!(values, vec![6.0, 12.0, 18.0]);

        rt.shutdown_background();
    }

    /// A single server hosting multiple devices: two indices on the same address resolve to
    /// two distinct sessions (distinct interpreters/runner threads). Moving a tensor between
    /// them exercises the multi-device path within one host.
    #[test]
    pub fn test_multi_device_single_server() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        // One server, two devices.
        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![
                Default::default(),
                Default::default(),
            ])
            .port(3030)
            .start_async(),
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        let device_0 = Device::remote("ws://localhost:3030", 0);
        let device_1 = Device::remote("ws://localhost:3030", 1);

        // Distinct indices on the same address must be distinct devices.
        assert_ne!(device_0, device_1);

        let input_shape = [1, 28, 28];
        let input = Tensor::<3>::random(input_shape, Distribution::Default, &device_0);
        let numbers_expected: Vec<f32> = input.to_data().to_vec().unwrap();

        // Move tensor to the second device on the same host and back.
        let input = input.to_device(&device_1);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        let input = input.to_device(&device_0);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        rt.shutdown_background();
    }

    /// Concurrent multi-device regression (DDP-style): two user threads, each pinned to a device,
    /// running simultaneously and each iteration moving a tensor to the *other* device and back.
    ///
    /// This used to deadlock because the client's transfer-id counter never persisted its
    /// increment (`TensorTransferId` is `Copy`, so the increment landed on a throwaway local).
    /// Every same-host transfer after the first reused the same id; sequentially that's harmless
    /// (each expose is taken before the next), but two transfers in flight at once then collided
    /// in the server's `local_comm` rendezvous and one `take` hung forever. Single-threaded
    /// workloads never tripped it — `into_data` serializes each iteration — so the bug only shows
    /// up under genuine concurrency.
    #[test]
    fn test_multi_device_concurrent_to_device_deadlock() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![
                Default::default(),
                Default::default(),
            ])
            .port(3060)
            .start_async(),
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        with_deadlock_watchdog(std::time::Duration::from_secs(30), || {
            let device0 = Device::remote("ws://localhost:3060", 0);
            let device1 = Device::remote("ws://localhost:3060", 1);

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

    /// `Device::enumerate(DeviceType::remote(addr))` lists every device the server hosts, by
    /// connecting once and reading the device count off the init handshake.
    #[test]
    pub fn test_enumerate_remote_devices() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        // One server hosting three devices.
        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![
                Default::default(),
                Default::default(),
                Default::default(),
            ])
            .port(3040)
            .start_async(),
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        let devices = Device::enumerate(DeviceType::remote("ws://localhost:3040")).into_vec();

        // The server reports its three devices, in index order.
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0], Device::remote("ws://localhost:3040", 0));
        assert_eq!(devices[1], Device::remote("ws://localhost:3040", 1));
        assert_eq!(devices[2], Device::remote("ws://localhost:3040", 2));

        // Distinct indices are distinct devices.
        assert_ne!(devices[0], devices[1]);
        assert_ne!(devices[1], devices[2]);

        // The enumerated devices are usable: run an op on the last one.
        let input = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &devices[2]);
        let numbers: Vec<f32> = (input * 2.0).to_data().to_vec().unwrap();
        assert_eq!(numbers, vec![2.0, 4.0, 6.0, 8.0]);

        rt.shutdown_background();
    }

    /// Exercises the cross-backend transfer body: local tensor → remote (data round-trip
    /// through `TensorData`), an op on the remote, then remote → local.
    /// Run `body` on a worker thread and report whether it finished within `timeout`.
    ///
    /// Unlike [`with_deadlock_watchdog`], a panic inside `body` counts as "finished": these error
    /// tests assert that a failure *surfaces* (as an `Err` or a panic) instead of hanging, so all
    /// that matters is the thread came back. Returns `false` if it was still running at the
    /// deadline — i.e. the call hung.
    fn finishes_within(timeout: std::time::Duration, body: impl FnOnce() + Send + 'static) -> bool {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Swallow panics: a disconnected read panicking on the error path is an acceptable
            // "didn't hang" outcome and must not abort the whole test process.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
            let _ = tx.send(());
        });
        rx.recv_timeout(timeout).is_ok()
    }

    /// When the server goes down, a client call that awaits a response (here a tensor read) must
    /// fail promptly instead of blocking forever. Before the fix the response-demux task just
    /// exited on the closed stream, leaving every pending callback — and any later request —
    /// parked on a oneshot that would never be completed.
    #[test]
    fn test_server_down_does_not_hang_client() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3070)
                .start_async(),
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        let device = Device::remote("ws://localhost:3070", 0);

        // One successful round-trip so the sockets are actually up and the demux task is running.
        let input = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let warmup: Vec<f32> = (input * 2.0).to_data().to_vec().unwrap();
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

    /// A client that disconnects abruptly mid-session (socket dropped, no `Close`) must not wedge
    /// the server: it should clean the session up and keep serving everyone else. We drive a raw
    /// connection here because a `Device`'s client is process-cached and never dropped mid-test.
    #[test]
    fn test_client_disconnect_handled_cleanly_by_server() {
        use crate::shared::{RemoteMessage, SessionId, Task};
        use burn_communication::{CommunicationChannel, Message, Protocol, ProtocolClient};
        use std::str::FromStr;

        type Client = <crate::shared::RemoteProtocol as Protocol>::Client;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3090)
                .start_async(),
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        // Raw client: connect a submit stream, init a session and send one task so the server
        // spawns the session worker, then drop the socket without a `Close` to mimic a crash.
        {
            let rtc = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let address = burn_communication::Address::from_str("ws://localhost:3090").unwrap();
            let session_id = SessionId::new();

            rtc.block_on(async {
                let mut submit = Client::connect(address, "submit")
                    .await
                    .expect("raw submit connect");

                let frame = |msgs: Vec<RemoteMessage>| -> Message {
                    Message::new(rmp_serde::to_vec(&msgs).unwrap().into())
                };

                submit
                    .send(frame(vec![RemoteMessage::Init(session_id, 0)]))
                    .await
                    .expect("send init");
                submit
                    .send(frame(vec![RemoteMessage::Task(Task::Seed(0))]))
                    .await
                    .expect("send task");
                // Drop `submit` here (end of block): the server sees the stream end without a
                // `Close` and must run the cleanup path.
            });
        }

        // Give the server a moment to tear the abandoned session down.
        std::thread::sleep(std::time::Duration::from_millis(500));

        // The server must have survived: a fresh, normal client on the same server still works.
        let finished = finishes_within(std::time::Duration::from_secs(10), || {
            let device = Device::remote("ws://localhost:3090", 0);
            let input = Tensor::<2>::from_floats([[10.0, 20.0]], &device);
            let numbers: Vec<f32> = (input * 3.0).to_data().to_vec().unwrap();
            assert_eq!(numbers, vec![30.0, 60.0]);
        });
        assert!(
            finished,
            "server stopped serving after a client disconnected abruptly"
        );

        rt.shutdown_timeout(std::time::Duration::from_millis(100));
    }

    #[test]
    pub fn test_to_device_local_to_remote() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3020)
                .start_async(),
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        let local = Device::default();
        let remote = Device::remote("ws://localhost:3020", 0);

        // Create on local, move to remote.
        let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &local);
        let on_remote = input.clone().to_device(&remote);

        // Run an op while on the remote.
        let doubled = on_remote * 2.0;

        // Move back to local and verify.
        let back = doubled.to_device(&local);
        let numbers: Vec<f32> = back.to_data().to_vec().unwrap();
        assert_eq!(numbers, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        rt.shutdown_background();
    }
}

#[cfg(all(test, feature = "fusion", feature = "server"))]
mod fusion_tests {
    use crate::{RemoteBackend, RemoteDevice, client::RemoteChannel, shared::RemoteProtocol};
    use burn_backend::{Backend, Shape, TensorData};
    use burn_communication::Protocol;
    use burn_router::BackendRouter;

    // `RemoteBackend` is `Fusion<PlainRemote>` under the `fusion` feature; `PlainRemote` is the
    // unwrapped router backend, used as the reference to compare against.
    type PlainRemote = BackendRouter<RemoteChannel<<RemoteProtocol as Protocol>::Client>>;

    fn input() -> TensorData {
        TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]])
    }

    /// Run the same small multi-op graph `iters` times on backend `B`, reading the result each
    /// iteration. Every iteration has an identical op structure, so a fusion backend registers the
    /// optimization once and replays it by id on the later iterations.
    ///
    /// The graph deliberately includes a reshape, so one of the intermediate tensors (`c`) has a
    /// *different* shape than the inputs/outputs — exercising the server's reconstruction of
    /// intermediate shapes from the shape-dim map (rather than them being sent per replay).
    fn run<B: Backend<Device = RemoteDevice>>(
        device: &RemoteDevice,
        iters: usize,
    ) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for _ in 0..iters {
            let a = B::float_from_data(input(), device); // [2, 3]
            let b = B::float_exp(a); // [2, 3]
            let c = B::float_reshape(b, Shape::from([3, 2])); // [3, 2] intermediate (distinct shape)
            let d = B::float_log(c); // [3, 2]
            let data = burn_std::reader::try_read_sync(B::float_into_data(d))
                .expect("remote read should resolve synchronously")
                .expect("read should succeed");
            out.push(data.to_vec::<f32>().unwrap());
        }
        out
    }

    /// The fusion-enabled remote backend must produce exactly the same results as the plain remote
    /// backend across a repeated computation that exercises register-once + replay-by-id.
    #[test]
    fn fusion_matches_plain_remote() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<burn_flex::Flex>::new(vec![Default::default()])
                .port(3100)
                .start_async(),
        );
        rt.spawn(
            crate::server::RemoteServerBuilder::<burn_flex::Flex>::new(vec![Default::default()])
                .port(3110)
                .start_async(),
        );
        std::thread::sleep(std::time::Duration::from_millis(500));

        let plain_device = RemoteDevice::new("ws://localhost:3100", 0);
        let fused_device = RemoteDevice::new("ws://localhost:3110", 0);

        let iters = 5;
        let expected = run::<PlainRemote>(&plain_device, iters);
        let actual = run::<RemoteBackend>(&fused_device, iters);

        assert_eq!(actual.len(), iters);
        assert_eq!(expected.len(), iters);
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_eq!(a.len(), e.len());
            for (av, ev) in a.iter().zip(e.iter()) {
                assert!(
                    (av - ev).abs() < 1e-5,
                    "fusion result {av} differs from plain remote {ev}"
                );
            }
        }

        rt.shutdown_background();
    }

    /// A *source* custom op (no tensor inputs — it builds a tensor from scalars on the server) whose
    /// output is then consumed by a follow-up op, read back, repeated to exercise register-once +
    /// replay. This mirrors the server-side data-loader extension pattern and isolates it from the
    /// training stack — if the fusion graph mishandles a source custom op's outputs, it surfaces here
    /// as a "Should have handle for tensor ..." panic on the server.
    #[test]
    fn fusion_custom_source_op_then_followup() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3120)
                .custom_op("make_floats", |handles, ir, device| {
                    // Build a 1-D float tensor from the op's scalars — a pure source (no inputs).
                    let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                    let n = values.len();
                    let tensor = Flex::float_from_data(TensorData::new(values, [n]), device);
                    handles.register_float_tensor::<Flex>(&ir.outputs[0].id, tensor);
                })
                .start_async(),
        );
        std::thread::sleep(std::time::Duration::from_millis(500));

        let device = RemoteDevice::new("ws://localhost:3120", 0);

        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let out_ir =
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32);
            let made = client
                .register(CustomOpIr::with_scalars(
                    "make_floats",
                    &[],
                    &[out_ir],
                    vec![
                        ScalarIr::Float(1.0),
                        ScalarIr::Float(2.0),
                        ScalarIr::Float(3.0),
                    ],
                ))
                .output();

            // Follow-up op consuming the source output — forces a graph that references the custom
            // op's output as an input, the scenario that breaks during training.
            let doubled = <RemoteBackend as FloatTensorOps<RemoteBackend>>::float_exp(made);
            let data = burn_std::reader::try_read_sync(<RemoteBackend as FloatTensorOps<
                RemoteBackend,
            >>::float_into_data(doubled))
            .expect("remote read should resolve synchronously")
            .expect("read should succeed");

            let values = data.to_vec::<f32>().unwrap();
            let expected = [1.0f32.exp(), 2.0f32.exp(), 3.0f32.exp()];
            for (a, e) in values.iter().zip(expected.iter()) {
                assert!((a - e).abs() < 1e-4, "iter {i}: {a} vs {e}");
            }
        }

        rt.shutdown_background();
    }

    /// Read a source custom op's output *directly* (it is the boundary output, with no follow-up op
    /// consuming it). This is what the data loader does — `batch.tokens.to_data()` — and the case the
    /// other two tests don't cover (they always feed the output into another op first).
    #[test]
    fn fusion_read_source_output_directly() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3140)
                .custom_op("make_floats", |handles, ir, device| {
                    let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                    let n = values.len();
                    let tensor = Flex::float_from_data(TensorData::new(values, [n]), device);
                    handles.register_float_tensor::<Flex>(&ir.outputs[0].id, tensor);
                })
                .start_async(),
        );
        std::thread::sleep(std::time::Duration::from_millis(500));

        let device = RemoteDevice::new("ws://localhost:3140", 0);

        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let out_ir =
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32);
            let made = client
                .register(CustomOpIr::with_scalars(
                    "make_floats",
                    &[],
                    &[out_ir],
                    vec![
                        ScalarIr::Float(1.0),
                        ScalarIr::Float(2.0),
                        ScalarIr::Float(3.0),
                    ],
                ))
                .output();

            // Read the source output directly — no follow-up op.
            let data = burn_std::reader::try_read_sync(<RemoteBackend as FloatTensorOps<
                RemoteBackend,
            >>::float_into_data(made))
            .expect("remote read should resolve synchronously")
            .expect("read should succeed");
            let values = data.to_vec::<f32>().unwrap();
            assert_eq!(values, vec![1.0, 2.0, 3.0], "iter {i}");
        }

        rt.shutdown_background();
    }

    /// Closer to the data-loader: a source custom op with *three* outputs that are consumed at
    /// *different depths* of the following graph (one immediately, one mid-graph, one only at the
    /// end — like `tokens`/`mask`/`labels`). Exercises a source op's outputs surviving across many
    /// ops before being bound as inputs, under register-once + replay.
    #[test]
    fn fusion_custom_source_multi_output_long_lived() {
        use crate::client::CustomOpClient;
        use burn_backend::DType;
        use burn_backend::ops::FloatTensorOps;
        use burn_flex::Flex;
        use burn_ir::{CustomOpIr, OperationOutput, ScalarIr, TensorIr};

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(
            crate::server::RemoteServerBuilder::<Flex>::new(vec![Default::default()])
                .port(3130)
                .custom_op("make3", |handles, ir, device| {
                    // 9 scalars → three [3] outputs (chunks of 3).
                    let values: Vec<f32> = ir.scalars.iter().map(|s| s.elem::<f32>()).collect();
                    for (i, out) in ir.outputs.iter().enumerate() {
                        let chunk = values[i * 3..(i + 1) * 3].to_vec();
                        let tensor = Flex::float_from_data(TensorData::new(chunk, [3]), device);
                        handles.register_float_tensor::<Flex>(&out.id, tensor);
                    }
                })
                .start_async(),
        );
        std::thread::sleep(std::time::Duration::from_millis(500));

        let device = RemoteDevice::new("ws://localhost:3130", 0);

        for i in 0..5 {
            let client = CustomOpClient::new(&device);
            let mk = |client: &CustomOpClient| {
                TensorIr::uninit(client.create_empty_handle(), Shape::from([3]), DType::F32)
            };
            let [a, b, c] = client
                .register(CustomOpIr::with_scalars(
                    "make3",
                    &[],
                    &[mk(&client), mk(&client), mk(&client)],
                    (1..=9).map(|v| ScalarIr::Float(v as f64)).collect(),
                ))
                .outputs::<3>();

            // a consumed immediately, b mid-graph, c only at the end — so b and c are source-op
            // outputs that survive across several ops before being bound as inputs.
            type B = RemoteBackend;
            let t = <B as FloatTensorOps<B>>::float_exp(a);
            let t = <B as FloatTensorOps<B>>::float_add(t, b);
            let t = <B as FloatTensorOps<B>>::float_log(t);
            let t = <B as FloatTensorOps<B>>::float_add(t, c);
            let data =
                burn_std::reader::try_read_sync(<B as FloatTensorOps<B>>::float_into_data(t))
                    .expect("remote read should resolve synchronously")
                    .expect("read should succeed");

            let values = data.to_vec::<f32>().unwrap();
            // a=[1,2,3], b=[4,5,6], c=[7,8,9]; t = log(exp(a)+b) + c
            let expected: Vec<f32> = (0..3)
                .map(|k| {
                    let a = (k + 1) as f32;
                    let b = (k + 4) as f32;
                    let c = (k + 7) as f32;
                    (a.exp() + b).ln() + c
                })
                .collect();
            for (g, e) in values.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-3, "iter {i}: {g} vs {e}");
            }
        }

        rt.shutdown_background();
    }
}
