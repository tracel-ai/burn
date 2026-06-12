#[cfg(feature = "client")]
pub(crate) mod client;

#[cfg(feature = "server")]
pub mod server;

pub(crate) mod shared;

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
    /// `BackendIr`, call [`server::start_websocket`] directly with the
    /// concrete backend type parameter.
    pub type RemoteBackend = BackendRouter<RemoteChannel<<RemoteProtocol as Protocol>::Client>>;

    pub use client::RemoteDevice;
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

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default()],
            3000,
        ));
        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default()],
            3010,
        ));

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
        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default(), Default::default()],
            3030,
        ));

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

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default(), Default::default()],
            3060,
        ));

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
        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default(), Default::default(), Default::default()],
            3040,
        ));

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

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default()],
            3070,
        ));

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

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default()],
            3090,
        ));

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

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            vec![Default::default()],
            3020,
        ));

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
