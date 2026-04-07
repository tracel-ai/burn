use std::marker::PhantomData;

use burn_backend::{
    DeviceId,
    distributed::{DistributedBackend, ReduceOperation},
    tensor::{Device, FloatTensor},
};
use burn_ir::{AllReduceOpIr, BaseOperationIr, HandleContainer, OperationIr};

use crate::{
    Fusion, FusionBackend,
    stream::{Operation, OperationStreams},
};
use burn_ir::OperationOutput;

impl<B: FusionBackend + DistributedBackend> DistributedBackend for Fusion<B> {
    // unsafe fn all_reduce(
    //     tensors: Vec<FloatTensor<Self>>,
    //     op: ReduceOperation,
    // ) -> Vec<FloatTensor<Self>> {
    //     #[derive(new, Debug)]
    //     struct AllReduceOps<B: FusionBackend> {
    //         desc: AllReduceOpIr,
    //         op: ReduceOperation,
    //         _b: PhantomData<B>,
    //     }

    //     impl<B: FusionBackend + DistributedBackend> Operation<B::FusionRuntime> for AllReduceOps<B> {
    //         fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
    //             let tensors = self
    //                 .desc
    //                 .tensors
    //                 .iter()
    //                 .map(|tensor| handles.get_float_tensor::<B>(tensor))
    //                 .collect();

    //             let outputs = unsafe { B::all_reduce(tensors, self.op) };

    //             self.desc
    //                 .out
    //                 .iter()
    //                 .zip(outputs)
    //                 .for_each(|(out, result)| handles.register_float_tensor::<B>(&out.id, result));
    //         }
    //     }

    //     const num_tensors: usize = tensors.len();
    //     let streams = OperationStreams::with_inputs(&tensors);

    //     let client = tensors.first().unwrap().client.clone();
    //     let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
    //     let desc = AllReduceOpIr::create(tensors, || client.create_empty_handle());

    //     client
    //         .register(
    //             streams,
    //             OperationIr::BaseFloat(BaseOperationIr::AllReduce(desc.clone())),
    //             AllReduceOps::<B>::new(desc, op),
    //         )
    //         .outputs::<num_tensors>()
    //         .into()
    // }

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
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = AllReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::AllReduce(desc.clone())),
                AllReduceOps::<B>::new(desc, op, device_ids),
            )
            .output()
            .into()
    }

    fn sync_collective(device: &Device<Self>) {
        todo!()
    }
}
