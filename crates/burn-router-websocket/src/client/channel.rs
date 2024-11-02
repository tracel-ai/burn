use burn_router::{RouterTensor, RunnerChannel, TensorHandle};
use burn_tensor::repr::TensorDescription;

use super::{
    router::{WsBridge, WsDevice},
    WsClient,
};

/// A local channel with direct connection to the backend runner clients.
#[derive(Clone)]
pub struct WsChannel;

impl RunnerChannel for WsChannel {
    type Device = WsDevice;
    type Bridge = WsBridge;
    type Client = WsClient;

    type FloatElem = f32;

    type IntElem = i32;

    fn name() -> String {
        "http".into()
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        WsClient::init(device.clone())
    }

    fn get_tensor_handle(
        _tensor: &TensorDescription,
        _client: &Self::Client,
    ) -> TensorHandle<Self::Bridge> {
        panic!("Unsupported")
    }

    fn register_tensor(
        _client: &Self::Client,
        _handle: TensorHandle<Self::Bridge>,
        _shape: Vec<usize>,
        _dtype: burn_tensor::DType,
    ) -> RouterTensor<Self::Client> {
        panic!("Unsupported")
    }
}
