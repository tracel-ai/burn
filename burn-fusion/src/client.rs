use crate::{
    channel::FusionChannel,
    graph::{FusedBackend, GraphExecution, TensorOps},
};
use std::{marker::PhantomData, sync::Arc};

pub struct FusionClient<B, C, G> {
    channel: Arc<C>,
    _backend: PhantomData<B>,
    _graph: PhantomData<G>,
}

impl<B, C, G> Clone for FusionClient<B, C, G> {
    fn clone(&self) -> Self {
        Self {
            channel: self.channel.clone(),
            _backend: PhantomData,
            _graph: PhantomData,
        }
    }
}

impl<B, G, C> FusionClient<B, C, G>
where
    B: FusedBackend,
    G: GraphExecution<B>,
    C: FusionChannel<B, G>,
{
    pub fn new(channel: C) -> Self {
        Self {
            channel: Arc::new(channel),
            _backend: PhantomData,
            _graph: PhantomData,
        }
    }
    pub fn register(&self, ops: TensorOps<B::FloatElem, B::IntElem>) {
        self.channel.register(ops);
    }

    pub fn sync(&self) {
        self.channel.sync();
    }
}
