use burn_backend::{
    DeviceId, DeviceOps, StreamId, TensorMetadata,
    distributed::{DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};

use crate::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement, ops::empty};

impl<R, F, I, BT> DistributedBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    // unsafe fn all_reduce(
    //     tensors: Vec<FloatTensor<Self>>,
    //     op: ReduceOperation,
    // ) -> Vec<FloatTensor<Self>> {
    //     let all_ids = tensors.iter().map(|t| t.device.id()).collect::<Vec<_>>();
    //     let mut output = vec![];

    //     for tensor in tensors {
    //         let device = &tensor.device;
    //         let old = unsafe { StreamId::swap(tensor.handle.stream) };
    //         let out_tensor = empty(tensor.shape(), device, tensor.dtype());

    //         let op = match op {
    //             ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
    //             ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
    //         };

    //         let client = R::client(device);
    //         client.all_reduce(
    //             tensor.handle.clone(),
    //             out_tensor.handle.clone(),
    //             tensor.dtype.into(),
    //             all_ids.clone(),
    //             op,
    //         );

    //         output.push(out_tensor);

    //         unsafe { StreamId::swap(old) };
    //     }

    //     output
    // }

    unsafe fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> FloatTensor<Self> {
        let device = &tensor.device;
        StreamId::executes(tensor.handle.stream, || {
            log::info!("create empty");
            let out_tensor = empty(tensor.shape(), device, tensor.dtype());

            let op = match op {
                ReduceOperation::Sum => cubecl::server::ReduceOperation::Sum,
                ReduceOperation::Mean => cubecl::server::ReduceOperation::Mean,
            };

            log::info!("get client");
            let client = R::client(device);
            log::info!("all reduce");
            client.all_reduce(
                tensor.handle.clone(),
                out_tensor.handle.clone(),
                tensor.dtype.into(),
                device_ids.clone(),
                op,
            );
            log::info!("all reduce return");
            out_tensor
        })
        // let old = unsafe { StreamId::swap(tensor.handle.stream) };

        // unsafe { StreamId::swap(old) };
        // out_tensor
    }

    fn sync_collective(device: &Device<Self>) {
        let client = R::client(device);
        client.sync_collective();
    }
}
