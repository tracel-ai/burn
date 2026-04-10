use std::marker::PhantomData;

use burn_backend::{
    DeviceId, DeviceOps,
    distributed::{DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};
use burn_ir::{AllReduceOpIr, BaseOperationIr, HandleContainer, OperationIr};

use crate::{
    Fusion, FusionBackend, get_client,
    stream::{Operation, OperationStreams},
};
use burn_ir::OperationOutput;

impl<B: FusionBackend + DistributedBackend> DistributedBackend for Fusion<B> {
    unsafe fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> FloatTensor<Self> {
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
                let output = unsafe { B::all_reduce(tensor, self.op, self.device_ids.clone()) };
                println!("fusion submitted all_reduce");
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        println!("fusion all_reduce");

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = AllReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        let out = client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::AllReduce(desc.clone())),
                AllReduceOps::<B>::new(desc, op, device_ids),
            )
            .output()
            .into();

        client.drain();
        client.flush_queue();

        out
    }

    fn sync_collective(device: &Device<Self>) {
        println!("fusion sync_collective: {:?}", device.id());
        let client = get_client::<B>(device);
        println!("fusion sync_collective got client: {:?}", device.id());
        client.sync_collective::<B>(device.clone());
    }
}
