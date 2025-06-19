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

    use burn_router::BackendRouter;
    use client::WsChannel;

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
    pub type RemoteBackend = BackendRouter<WsChannel>;

    pub use client::WsDevice as RemoteDevice;
}
#[cfg(feature = "client")]
pub use __client::*;

#[cfg(all(test, feature = "client", feature = "server"))]
mod tests {
    use crate::{client::WsChannel, RemoteBackend};
    use burn_ndarray::NdArray;
    use burn_router::drop_client;
    use burn_tensor::{Distribution, Tensor};
    use serial_test::serial;

    #[test]
    #[serial]
    pub fn test_to_device_over_websocket() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        rt.spawn(crate::server::start_async::<NdArray>(
            Default::default(),
            3000,
        ));
        rt.spawn(crate::server::start_async::<NdArray>(
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

        drop_client::<WsChannel>(&remote_device_1);
        drop_client::<WsChannel>(&remote_device_2);

        rt.shutdown_background();
    }

    #[test]
    #[serial]
    pub fn test_to_device_over_websocket_add() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        rt.spawn(crate::server::start_async::<NdArray>(
            Default::default(),
            3000,
        ));
        rt.spawn(crate::server::start_async::<NdArray>(
            Default::default(),
            3010,
        ));

        let remote_device_1 = super::RemoteDevice::new("ws://localhost:3000");
        let remote_device_2 = super::RemoteDevice::new("ws://localhost:3010");

        // Some random input
        let input_shape = [1, 28, 28];
        let tensor_1 = Tensor::<RemoteBackend, 3>::random(
            input_shape,
            Distribution::Default,
            &remote_device_1,
        );
        let numbers_1: Vec<f32> = tensor_1.to_data().to_vec().unwrap();
        let tensor_2 = Tensor::<RemoteBackend, 3>::random(
            input_shape,
            Distribution::Default,
            &remote_device_2,
        );
        let numbers_2: Vec<f32> = tensor_2.to_data().to_vec().unwrap();
        let sum_expected: Vec<f32> = numbers_1.iter().enumerate().map(|(i, x)| x + numbers_2[i]).collect();

        // Move tensor 1 to device 2
        let tensor_1_on_2 = tensor_1.clone().to_device(&remote_device_2);
        let sum_on_2 = tensor_2.add(tensor_1_on_2);


        // Move tensor back to device 1
        let sum_on_1 = sum_on_2.to_device(&remote_device_1);
        let sum_actual: Vec<f32> = sum_on_1.to_data().to_vec().unwrap();

        for (i, &x) in sum_actual.iter().enumerate() {
            assert_eq!(x, sum_expected[i]);
        }

        // find a way to shut down the Client which has the Sender which will close the channel to the workers which are hanging threads.        

        drop_client::<WsChannel>(&remote_device_1);
        drop_client::<WsChannel>(&remote_device_2);

        rt.shutdown_background();
    }
}
