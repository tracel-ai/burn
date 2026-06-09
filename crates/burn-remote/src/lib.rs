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
