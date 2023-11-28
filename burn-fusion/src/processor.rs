use crate::{
    graph::{Ops, Optimization, TensorOpsDescription},
    FusionBackend,
};

pub trait OpsProcessor<B: FusionBackend> {
    fn process(&mut self, ops_desc: TensorOpsDescription, ops: Box<dyn Ops<B>>);
    fn sync(&mut self);
}

pub struct FusionOpsBuilderProcessor<B: FusionBackend> {
    optimizations: Vec<Optimization<B>>,
}

impl<B: FusionBackend> OpsProcessor<B> for FusionOpsBuilderProcessor<B> {
    fn process(&mut self, ops_desc: TensorOpsDescription, ops: Box<dyn Ops<B>>) {
        todo!()
    }
    fn sync(&mut self) {}
}
