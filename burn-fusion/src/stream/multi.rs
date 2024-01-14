use super::{
    execution::{ExecutionMode, Process, Processor},
    store::{ExplorationId, ExplorationStore},
    Operation, OperationDescription, OperationQueue,
};
use crate::{FusionBackend, HandleContainer};

/// Keep track of multiple concurrent streams of operations.
///
/// TODO: Actually support multiple streams.
pub struct MultiStream<B: FusionBackend> {
    streams: Vec<Stream<B>>,
    optimizations: ExplorationStore<B::Optimization>,
}

struct Stream<B: FusionBackend> {
    queue: OperationQueue<B>,
    processor: Processor<B::Optimization>,
}

#[derive(new)]
struct StreamExecutionProcess<'a, B: FusionBackend> {
    queue: &'a mut OperationQueue<B>,
    handles: &'a mut HandleContainer<B>,
}

impl<'i, B: FusionBackend> Process<B::Optimization> for StreamExecutionProcess<'i, B> {
    fn operations<'a>(&'a self) -> &[OperationDescription] {
        &self.queue.relative
    }

    fn execute(&mut self, id: ExplorationId, store: &mut ExplorationStore<B::Optimization>) {
        self.queue.execute(id, self.handles, store)
    }
}

impl<B: FusionBackend> MultiStream<B> {
    pub(crate) fn new(device: B::FusionDevice) -> Self {
        Self {
            streams: vec![Stream::new(device)],
            optimizations: ExplorationStore::new(),
        }
    }

    /// Register a new tensor operation.
    pub fn register(
        &mut self,
        desc: OperationDescription,
        operation: Box<dyn Operation<B>>,
        handles: &mut HandleContainer<B>,
    ) {
        // TODO: Support more than only one stream.
        if let Some(item) = self.streams.first_mut() {
            item.queue.add(desc, operation);
            item.processor.process(
                StreamExecutionProcess::new(&mut item.queue, handles),
                &mut self.optimizations,
                ExecutionMode::Lazy,
            );
        };
    }

    /// Drain the streams.
    pub fn drain(&mut self, handles: &mut HandleContainer<B>) {
        self.streams.iter_mut().for_each(|item| {
            item.processor.process(
                StreamExecutionProcess::new(&mut item.queue, handles),
                &mut self.optimizations,
                ExecutionMode::Sync,
            );
        });
    }
}

impl<B: FusionBackend> Stream<B> {
    fn new(device: B::FusionDevice) -> Self {
        Self {
            processor: Processor::new(B::optimizations(device.into())),
            queue: OperationQueue::new(),
        }
    }
}
