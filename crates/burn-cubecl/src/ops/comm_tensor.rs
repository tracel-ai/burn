use burn_backend::ops::TensorRef;
use burn_backend::{DeviceId, DeviceOps};
use burn_backend::{ReduceOperation, ops::CommunicationTensorOps, tensor::Device};

use crate::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

impl<R, F, I, BT> CommunicationTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    // TODO: actually just cuda
    fn supports_native_collective(_device: &Device<Self>) -> bool {
        true
    }

    fn all_reduce_in_place_native(
        tensor: TensorRef<Self>,
        _peer_id: burn_backend::PeerId,
        all_ids: Vec<burn_backend::PeerId>,
        op: ReduceOperation,
    ) {
        unsafe {
            let tensor = &**tensor.0;
            let device = &tensor.device;
            let client = R::client(device);
            let mut all_ids = all_ids.iter().map(|p| p.0).collect::<Vec<u32>>();
            all_ids.sort();
            let all_ids = all_ids
                .iter()
                .map(|val| DeviceId::new(device.id().type_id, *val))
                .collect();
            println!("{all_ids:?}");
            client.all_reduce(
                tensor.handle.clone(),
                tensor.handle.clone(),
                tensor.dtype.into(),
                all_ids,
                op.into(),
            );
        }
    }

    fn collective_sync_native(device: &Device<Self>) {
        println!(
            "[{:?}] sync native: R::client - {:?}",
            std::thread::current().id(),
            device.id()
        );
        let client = R::client(&device);
        println!(
            "[{:?}] sync native: R::client DONE - {:?}",
            std::thread::current().id(),
            device.id()
        );
        client.sync_collective();
    }
}
