use std::marker::PhantomData;

use burn_backend::{
    DeviceId,
    distributed::{CollectiveTensor, DistributedOps, ReduceOperation},
    tensor::{Device, FloatTensor},
};
use burn_ir::{AllReduceOpIr, DistributedOperationIr, HandleContainer, OperationIr};

use crate::{
    Fusion, FusionBackend, get_client,
    stream::{Operation, StreamId},
};
use burn_ir::OperationOutput;

impl<B: FusionBackend> DistributedOps<Self> for Fusion<B> {
    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        #[derive(new, Debug)]
        struct AllReduceOps<B: FusionBackend> {
            desc: AllReduceOpIr,
            op: ReduceOperation,
            device_ids: Vec<DeviceId>,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for AllReduceOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let output = B::all_reduce(tensor, self.op, self.device_ids.clone());
                handles.register_float_tensor::<B>(&self.desc.out.id, unsafe {
                    output.assume_resolved()
                });
            }
        }

        let streams = StreamId::current();

        let client = tensor.client.clone();
        let desc = AllReduceOpIr::create(
            tensor.into_ir(),
            op,
            device_ids.iter().map(|id| (*id).into()).collect(),
            || client.create_empty_handle(),
        );

        let output = client
            .register(
                streams,
                OperationIr::Distributed(DistributedOperationIr::AllReduce(desc.clone())),
                AllReduceOps::<B>::new(desc, op, device_ids.clone()),
            )
            .output();

        client.ensure_collective_init::<B>(device_ids);

        CollectiveTensor::new(output)
    }

    fn sync_collective(device: &Device<Self>) {
        let client = get_client::<B>(device);
        client.sync_collective::<B>(device);
    }
}
