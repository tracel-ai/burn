use burn_backend::Shape;
use burn_ir::TensorIr;
use burn_router::{RouterChannel, RouterTensor, get_client};

use super::{
    RemoteClient,
    runner::{RemoteBridge, RemoteDevice, RemoteTensorHandle},
};

/// A local channel with direct connection to the backend runner clients.
pub struct RemoteChannel;

impl RouterChannel for RemoteChannel {
    type Device = RemoteDevice;
    type Bridge = RemoteBridge;
    type Client = RemoteClient;

    fn name(device: &Self::Device) -> String {
        format!("remote-{device:?}")
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        RemoteClient::init(device.clone())
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> RemoteTensorHandle {
        RemoteTensorHandle {
            client: client.clone(),
            tensor: tensor.clone(),
        }
    }

    fn register_tensor(
        _client: &Self::Client,
        _handle: RemoteTensorHandle,
        _shape: Shape,
        _dtype: burn_backend::DType,
    ) -> RouterTensor<Self::Client> {
        // This function is normally only used to move a tensor from a device to another.
        //
        // In other words, to change the client.
        panic!("Can't register manually a tensor on a remote channel.");
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
        let router_tensor: RouterTensor<RemoteClient> =
            RouterTensor::new(id, handle.tensor.shape, handle.tensor.dtype, target_client);

        router_tensor
    }
}

impl Clone for RemoteChannel {
    fn clone(&self) -> Self {
        RemoteChannel
    }
}
