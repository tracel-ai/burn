use core::marker::PhantomData;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use burn_common::stub::Mutex;

use crate::{
    backend::Backend,
    repr::{HandleContainer, ReprBackend},
    runner::{Runner, RunnerBackend, RunnerClient},
};

// In-memory runner client.
#[derive(Clone)]
pub struct InMemory<B: RunnerBackend> {
    _b: PhantomData<B>,
    // TODO: replace with `Runner`
    // handles: Arc<Mutex<HandleContainer<B::Handle>>>,
    runner: Runner<B>,
}

impl<B: RunnerBackend> RunnerClient for InMemory<B> {
    fn register(
        &self,
        op: crate::repr::OperationDescription,
        stream: burn_common::stream::StreamId,
    ) {
        // TODO: call runner.execute(op, stream)
        self.runner.execute(op, stream)
        // todo!()
    }

    fn read_tensor(
        &self,
        tensor: crate::repr::TensorDescription,
        stream: burn_common::stream::StreamId,
    ) -> impl core::future::Future<Output = crate::TensorData> + Send {
        // Clone any data we need from `self` or `tensor` that will be used in the async block
        let handles = self.handles.clone(); // Assuming `self.handles` is an `Arc<Mutex<...>>`
        let tensor = tensor.clone(); // Clone if necessary

        async move {
            let rank = tensor.shape.len();

            // Create a future for each possible rank
            let future = match rank {
                1 => B::float_into_data(handles.lock().unwrap().get_float_tensor::<B, 1>(&tensor)),
                // 2 => Box::pin(async move {
                //     B::float_into_data(handles.lock().unwrap().get_float_tensor::<B, 2>(&tensor))
                //         .await
                // }),
                _ => panic!("rank unsupported {rank}"),
            };

            // Await the future
            future.await
        }

        // TODO: should call runner x_into_data?
    }

    fn write_tensor(
        &self,
        data: crate::TensorData,
        stream: burn_common::stream::StreamId,
    ) -> crate::repr::TensorDescription {
        todo!()
    }

    fn empty_tensor(
        &self,
        shape: Vec<usize>,
        dtype: crate::DType,
        stream: burn_common::stream::StreamId,
    ) -> crate::repr::TensorDescription {
        todo!()
    }
}
