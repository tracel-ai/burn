use super::{
    execution::{ExecutionMode, StreamExecutor},
    optim::StreamOptimizations,
    Ops, Stream, TensorOpsDescription,
};
use crate::{FusionBackend, HandleContainer};

/// Keep track of multiple concurrent streams of operations.
///
/// TODO: Actually support multiple streams.
pub struct MultiStream<B: FusionBackend> {
    items: Vec<Item<B>>,
    optimizations: StreamOptimizations<B::Optimization>,
}

struct Item<B: FusionBackend> {
    stream: Stream<B>,
    executor: StreamExecutor<B>,
}

impl<B: FusionBackend> MultiStream<B> {
    pub(crate) fn new(device: B::FusionDevice) -> Self {
        Self {
            items: vec![Item::new(device)],
            optimizations: StreamOptimizations::new(),
        }
    }

    /// Register a new tensor operation.
    pub fn register(
        &mut self,
        ops_desc: TensorOpsDescription,
        ops: Box<dyn Ops<B>>,
        handles: &mut HandleContainer<B>,
    ) {
        // TODO: Support more than only one stream.
        self.items.first_mut().map(|item| {
            item.stream.add(ops_desc, ops);
            item.executor.execute(
                &mut item.stream,
                &mut self.optimizations,
                handles,
                ExecutionMode::Lazy,
            );
        });
    }

    /// Drain the streams.
    pub fn drain(&mut self, handles: &mut HandleContainer<B>) {
        self.items.iter_mut().for_each(|item| {
            item.executor.execute(
                &mut item.stream,
                &mut self.optimizations,
                handles,
                ExecutionMode::Sync,
            );
        });
    }
}

impl<B: FusionBackend> Item<B> {
    fn new(device: B::FusionDevice) -> Self {
        Self {
            executor: StreamExecutor::new(B::optimizations(device.into())),
            stream: Stream::new(),
        }
    }
}
