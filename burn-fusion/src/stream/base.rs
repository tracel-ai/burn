use super::{
    execution::{ExecutionMode, StreamExecutor},
    optim::StreamOptimizations,
    Ops, StreamDescription, TensorOpsDescription,
};
use crate::{FusionBackend, HandleContainer};

pub struct MultiStreams<B: FusionBackend> {
    streams: Vec<Stream<B>>,
    optimizations: StreamOptimizations<B::Optimization>,
}

struct Stream<B: FusionBackend> {
    executor: StreamExecutor<B>,
    description: StreamDescription<B>,
}

impl<B: FusionBackend> MultiStreams<B> {
    pub fn new(device: B::FusionDevice) -> Self {
        Self {
            streams: vec![Stream::new(device)],
            optimizations: StreamOptimizations::new(),
        }
    }

    pub fn register(
        &mut self,
        ops_desc: TensorOpsDescription,
        ops: Box<dyn Ops<B>>,
        handles: &mut HandleContainer<B>,
    ) {
        // TODO: Support more than only one stream.
        self.streams
            .first_mut()
            .map(|stream| stream.register(ops_desc, ops, &mut self.optimizations, handles));
    }

    pub fn drain_graph(&mut self, handles: &mut HandleContainer<B>) {
        self.streams
            .iter_mut()
            .for_each(|stream| stream.drain_graph(&mut self.optimizations, handles));
    }
}

impl<B: FusionBackend> Stream<B> {
    fn new(device: B::FusionDevice) -> Self {
        Self {
            executor: StreamExecutor::new(B::optimizations(&device.into())),
            description: StreamDescription::new(),
        }
    }

    fn register(
        &mut self,
        ops_desc: TensorOpsDescription,
        ops: Box<dyn Ops<B>>,
        optimizations: &mut StreamOptimizations<B::Optimization>,
        handles: &mut HandleContainer<B>,
    ) {
        self.description.add(ops_desc, ops);
        self.executor.execute(
            &mut self.description,
            optimizations,
            handles,
            ExecutionMode::NewOps,
        );
    }

    fn drain_graph(
        &mut self,
        optimizations: &mut StreamOptimizations<B::Optimization>,
        handles: &mut HandleContainer<B>,
    ) {
        // Check if we can execute.
        self.executor.execute(
            &mut self.description,
            optimizations,
            handles,
            ExecutionMode::Sync,
        );
    }
}
