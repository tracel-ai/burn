use burn_backend::{
    DeviceId, TensorMetadata,
    cubecl::dtype_to_elem_type,
    distributed::{CollectiveTensor, DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};

use crate::{
    CubeBackend, CubeRuntime,
    ops::numeric::{self, zeros_client},
};

impl<R: CubeRuntime> DistributedBackend for CubeBackend<R> {
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        println!("cube all_reduce");

        let device = &tensor.device.clone();
        let out_tensor = if tensor.handle.can_mut() && tensor.is_contiguous() {
            println!("canmut");
            tensor
        } else {
            println!("zeroo");
            let zeros_tensor = zeros_client::<R>(
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

        let mut client = R::client(device);
        println!("elemtype : {}", dtype_to_elem_type(out_tensor.dtype),);
        println!("out_tensor off start: {:?}", out_tensor.handle.offset_start);
        println!("out_tensor off end: {:?}", out_tensor.handle.offset_end);
        println!("out_tensor size: {}", out_tensor.handle.size_in_used());
        client.all_reduce(
            out_tensor.handle.clone(),
            out_tensor.handle.clone(),
            dtype_to_elem_type(out_tensor.dtype),
            device_ids.clone(),
            op,
        );
        CollectiveTensor::new(out_tensor)
    }

    fn sync_collective(device: &Device<Self>) {
        println!("cube sync_collective");

        let client = R::client(device);
        client.sync_collective();
    }
}
