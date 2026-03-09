use std::sync::mpsc::Receiver;

use crate::DeviceOps;
use crate::{Backend, PeerId, ReduceOperation, ops::TensorRef};

pub(crate) struct AllReduceArgs<B: Backend> {
    pub tensor: TensorRef<B>,
    pub device_ids: Vec<PeerId>,
}

#[derive(new)]
pub(crate) struct Worker<B: Backend> {
    receiver: Receiver<AllReduceArgs<B>>,
}

impl<B: Backend> Worker<B> {
    pub(crate) fn run(&self) {
        loop {
            let args = self
                .receiver
                .recv()
                .expect("Tensor communications worker channel hanged.");
            // TODO: sum hard coded.
            B::all_reduce_inplace_native(
                args.tensor.clone(),
                PeerId::from(B::comm_device(&args.tensor).id().index_id),
                args.device_ids.clone(),
                ReduceOperation::Sum,
            );
        }
    }
}
