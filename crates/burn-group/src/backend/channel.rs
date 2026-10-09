use burn_backend::{DType, Shape, TensorData};
use burn_ir::TensorIr;
use burn_router::{BackendRouter, MultiBackendBridge, RouterChannel, RouterClient, RouterTensor};
use burn_std::future::block_on;

use super::{GroupClient, GroupDevice};

/// A backend whose tensors are split over a [`GroupDevice`], each rank running the backend the
/// group was made with.
pub type GroupBackend = BackendRouter<GroupChannel>;

/// Routes a group's ops to its [`GroupClient`].
#[derive(Clone)]
pub struct GroupChannel;

impl RouterChannel for GroupChannel {
    type Device = GroupDevice;
    type Bridge = GroupBridge;
    type Client = GroupClient;

    fn name(device: &Self::Device) -> String {
        format!("group of {}", device.ranks())
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        GroupClient::new(*device)
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> TensorData {
        block_on(client.read_tensor_async(tensor.clone()))
            .unwrap_or_else(|error| panic!("Reading {} failed: {error:?}", tensor.id))
    }

    fn register_tensor(
        client: &Self::Client,
        handle: TensorData,
        _shape: Shape,
        _dtype: DType,
    ) -> RouterTensor<Self::Client> {
        client.register_tensor_data(handle)
    }
}

/// A tensor moving to another group travels as its whole value, replicated on arrival.
pub struct GroupBridge;

impl MultiBackendBridge for GroupBridge {
    type TensorHandle = TensorData;
    type Device = GroupDevice;

    fn change_backend_float(
        tensor: TensorData,
        _shape: Shape,
        _target: &GroupDevice,
    ) -> TensorData {
        tensor
    }

    fn change_backend_int(tensor: TensorData, _shape: Shape, _target: &GroupDevice) -> TensorData {
        tensor
    }

    fn change_backend_bool(tensor: TensorData, _shape: Shape, _target: &GroupDevice) -> TensorData {
        tensor
    }
}
