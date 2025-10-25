use std::marker::PhantomData;

use burn_communication::ProtocolClient;
use burn_ir::TensorIr;
use burn_router::{RouterTensor, RunnerChannel, get_client};
use burn_tensor::Shape;

use super::{
    RemoteClient,
    runner::{RemoteBridge, RemoteDevice, RemoteTensorHandle},
};

/// A local channel with direct connection to the backend runner clients.
pub struct RemoteChannel<C: ProtocolClient> {
    _p: PhantomData<C>,
}

impl<C: ProtocolClient> RunnerChannel for RemoteChannel<C> {
    type Device = RemoteDevice;
    type Bridge = RemoteBridge<C>;
    type Client = RemoteClient;

    type FloatElem = f32;

    type IntElem = i32;

    type BoolElem = u32;

    //type ComplexElem = burn_tensor::Complex32;

    fn name(device: &Self::Device) -> String {
        format!("remote-{device:?}")
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        RemoteClient::init::<C>(device.clone())
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> RemoteTensorHandle<C> {
        RemoteTensorHandle {
            client: client.clone(),
            tensor: tensor.clone(),
            _p: PhantomData,
        }
    }

    fn register_tensor(
        _client: &Self::Client,
        _handle: RemoteTensorHandle<C>,
        _shape: Shape,
        _dtype: burn_tensor::DType,
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

impl<C: ProtocolClient> Clone for RemoteChannel<C> {
    fn clone(&self) -> Self {
        RemoteChannel { _p: PhantomData }
    }
}
