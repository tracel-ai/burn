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
    use burn_tensor::{Device, Distribution, Tensor};

    #[test]
    pub fn test_to_device_over_websocket() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            Default::default(),
            3000,
        ));
        rt.spawn(crate::server::start_websocket_async::<Flex>(
            Default::default(),
            3010,
        ));

        // Give the servers a moment to bind before clients try to connect.
        std::thread::sleep(std::time::Duration::from_millis(500));

        let device_1 = Device::remote("ws://localhost:3000");
        let device_2 = Device::remote("ws://localhost:3010");

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

    /// Exercises the cross-backend transfer body: local tensor → remote (data round-trip
    /// through `TensorData`), an op on the remote, then remote → local.
    #[test]
    pub fn test_to_device_local_to_remote() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(crate::server::start_websocket_async::<Flex>(
            Default::default(),
            3020,
        ));

        std::thread::sleep(std::time::Duration::from_millis(500));

        let local = Device::default();
        let remote = Device::remote("ws://localhost:3020");

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
