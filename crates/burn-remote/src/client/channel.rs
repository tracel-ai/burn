use burn_ir::TensorIr;
use burn_router::{RouterTensor, RunnerChannel, RunnerClient, TensorHandle};

use super::{
    WsClient,
    runner::{WsBridge, WsDevice},
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

    type BoolElem = u32;

    fn name(device: &Self::Device) -> String {
        format!("remote-{device:?}")
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        WsClient::init(device.clone())
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> TensorHandle<Self::Bridge> {
        client.runtime.block_on(client.read_tensor(tensor.clone()))
    }

    fn register_tensor(
        client: &Self::Client,
        _handle: TensorHandle<Self::Bridge>,
        _shape: Vec<usize>,
        _dtype: burn_tensor::DType,
    ) -> RouterTensor<Self::Client> {
        let router_tensor = client.register_tensor_data(_handle);
        client.sync();

        router_tensor
    }
}
