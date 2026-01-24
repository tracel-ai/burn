#![cfg_attr(docsrs, feature(doc_cfg))]

#[macro_use]
extern crate derive_new;

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
    ///
    /// ```rust, ignore
    /// fn main() {
    ///     let device = Default::default();
    ///     let port = 3000;
    ///
    ///     // You need to activate the `server` feature flag to have access to this function.
    ///     burn::server::start::<burn::backend::Wgpu>(device, port);
    /// }
    ///```
    pub type RemoteBackend = BackendRouter<RemoteChannel<<RemoteProtocol as Protocol>::Client>>;

    pub use client::RemoteDevice;
}
#[cfg(feature = "client")]
pub use __client::*;

#[cfg(all(test, feature = "client", feature = "server"))]
mod tests {
    use crate::RemoteBackend;
    use burn_ndarray::NdArray;
    use burn_tensor::{Distribution, Tensor};

    #[test]
    pub fn test_to_device_over_websocket() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .build()
            .unwrap();

        rt.spawn(crate::server::start_websocket_async::<NdArray>(
            Default::default(),
            3000,
        ));
        rt.spawn(crate::server::start_websocket_async::<NdArray>(
            Default::default(),
            3010,
        ));

        let remote_device_1 = super::RemoteDevice::new("ws://localhost:3000");
        let remote_device_2 = super::RemoteDevice::new("ws://localhost:3010");

        // Some random input
        let input_shape = [1, 28, 28];
        let input = Tensor::<RemoteBackend, 3>::random(
            input_shape,
            Distribution::Default,
            &remote_device_1,
        );
        let numbers_expected: Vec<f32> = input.to_data().to_vec().unwrap();

        // Move tensor to device 2
        let input = input.to_device(&remote_device_2);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        // Move tensor back to device 1
        let input = input.to_device(&remote_device_1);
        let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
        assert_eq!(numbers, numbers_expected);

        rt.shutdown_background();
    }
}
