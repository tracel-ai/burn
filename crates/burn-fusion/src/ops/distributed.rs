use std::marker::PhantomData;

use burn_backend::{
    DeviceId,
    distributed::{CollectiveTensor, DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};
use burn_ir::{AllReduceOpIr, DistributedOperationIr, HandleContainer, OperationIr};

use crate::{
    Fusion, FusionBackend, get_client,
    stream::{Operation, OperationStreams},
};
use burn_ir::OperationOutput;

impl<B: FusionBackend + DistributedBackend> DistributedBackend for Fusion<B> {
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

        impl<B: FusionBackend + DistributedBackend> Operation<B::FusionRuntime> for AllReduceOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let output = B::all_reduce(tensor, self.op, self.device_ids.clone());
                handles.register_float_tensor::<B>(&self.desc.out.id, unsafe {
                    output.assume_resolved()
                });
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = AllReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        let output = client
            .register(
                streams,
                OperationIr::Distributed(DistributedOperationIr::AllReduce(desc.clone())),
                AllReduceOps::<B>::new(desc, op, device_ids.clone()),
            )
            .output()
            .into();

        client.ensure_collective_init::<B>(device_ids);

        CollectiveTensor::new(output)
    }

    fn sync_collective(device: &Device<Self>) {
        let client = get_client::<B>(device);
        client.sync_collective::<B>(device);
    }
}
