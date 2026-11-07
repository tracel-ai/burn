use burn_backend::{DType, Shape, TensorData};
use burn_ir::TensorIr;
use burn_router::{BackendRouter, MultiBackendBridge, RouterChannel, RouterClient, RouterTensor};
use burn_std::future::block_on;

use super::{ShardClient, ShardedComputeDevice};

/// A backend whose tensors can be split over a [`ShardedComputeDevice`], each member running the backend
/// the sharded compute device was made with.
pub type ShardBackend = BackendRouter<ShardChannel>;

/// Routes a sharded compute device's ops to its [`ShardClient`].
#[derive(Clone)]
pub struct ShardChannel;

impl RouterChannel for ShardChannel {
    type Device = ShardedComputeDevice;
    type Bridge = ShardBridge;
    type Client = ShardClient;

    fn name(_device: &Self::Device) -> String {
        "sharded compute device".into()
    }

    /// Every sharded compute device registers its client when made, so a device without one names no sharded compute device.
    fn init_client(device: &Self::Device) -> Self::Client {
        panic!(
            "{device:?} names no sharded compute device: make one with Device::shard_options or \
             ShardedComputeDevice::new"
        )
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

/// A tensor moving to another sharded compute device travels as its whole value, replicated on arrival.
pub struct ShardBridge;

impl MultiBackendBridge for ShardBridge {
    type TensorHandle = TensorData;
    type Device = ShardedComputeDevice;

    fn change_backend_float(
        tensor: TensorData,
        _shape: Shape,
        _target: &ShardedComputeDevice,
    ) -> TensorData {
        tensor
    }

    fn change_backend_int(
        tensor: TensorData,
        _shape: Shape,
        _target: &ShardedComputeDevice,
    ) -> TensorData {
        tensor
    }

    fn change_backend_bool(
        tensor: TensorData,
        _shape: Shape,
        _target: &ShardedComputeDevice,
        out_dtype: burn_backend::BoolDType,
    ) -> TensorData {
        tensor.convert_dtype(out_dtype.into())
    }
}
