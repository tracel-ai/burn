use burn_ir::TensorIr;
use burn_router::{RouterTensor, RunnerChannel, get_client};

use crate::shared::{ComputeTask, TensorRemote};

use super::{
    WsClient,
    runner::{RemoteTensorHandle, WsBridge, WsDevice},
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

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> RemoteTensorHandle {
        RemoteTensorHandle {
            client: client.clone(),
            tensor: tensor.clone(),
        }
    }

    fn register_tensor(
        client: &Self::Client,
        handle: RemoteTensorHandle,
        shape: Vec<usize>,
        dtype: burn_tensor::DType,
    ) -> RouterTensor<Self::Client> {
        let remote_tensor = TensorRemote {
            id: handle.tensor.id,
            address: client.device.address.to_string(),
        };
        let new_id = client.sender.new_tensor_id();

        client
            .sender
            .send(ComputeTask::RegisterTensorRemote(remote_tensor, new_id));

        RouterTensor::new(handle.tensor.id, shape, dtype, client.clone())
    }

    fn change_client_backend(
        tensor: RouterTensor<Self::Client>,
        target_device: &Self::Device, // target device
    ) -> RouterTensor<Self::Client> {
        // Get tensor handle from current client
        let original_client = tensor.client.clone();
        let desc = tensor.into_ir();
        let handle = Self::get_tensor_handle(&desc, &original_client);

        let handle = handle.change_backend(target_device);

        let id = handle.tensor.id;

        let target_client = get_client::<Self>(target_device);
        let router_tensor: RouterTensor<WsClient> =
            RouterTensor::new(id, handle.tensor.shape, handle.tensor.dtype, target_client);

        router_tensor
    }
}
