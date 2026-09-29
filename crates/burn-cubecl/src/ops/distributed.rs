use burn_backend::distributed::DistributedOps;

use crate::CubeBackend;

#[cfg(feature = "std")]
use crate::ops::numeric::{self, zeros_client};
#[cfg(feature = "std")]
use burn_backend::{
    DeviceId, TensorMetadata,
    cubecl::dtype_to_elem_type,
    distributed::{CollectiveTensor, ReduceOperation},
    tensor::{Device, FloatTensor},
};

impl DistributedOps<Self> for CubeBackend {
    #[cfg(feature = "std")]
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        let device = &tensor.device.clone();
        let out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
            tensor
        } else {
            let zeros_tensor = zeros_client(
                tensor.client.clone(),
                device.clone(),
                tensor.shape(),
                tensor.dtype(),
            );
            numeric::add(zeros_tensor, tensor)
        };

        let op = match op {
            ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
            ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
        };

        let mut client = device.client();
        client.all_reduce(
            out_tensor.handle.clone(),
            out_tensor.handle.clone(),
            dtype_to_elem_type(out_tensor.dtype),
            device_ids.clone(),
            op,
        );
        CollectiveTensor::new(out_tensor)
    }

    #[cfg(feature = "std")]
    fn sync_collective(device: &Device<Self>) {
        let client = device.client();
        client.sync_collective();
    }
}
