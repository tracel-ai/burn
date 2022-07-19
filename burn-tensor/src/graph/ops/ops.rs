use crate::{
    node::{NodeId, NodeStateRef},
    tape::Tape,
};

#[derive(new)]
pub struct BinaryRecordedState<'a, Lhs, Rhs, Out> {
    pub left: &'a NodeStateRef<Lhs>,
    pub right: &'a NodeStateRef<Rhs>,
    pub output: &'a NodeStateRef<Out>,
}

#[derive(new)]
pub struct SingleRecordedState<'a, In, Out> {
    pub input: &'a In,
    pub output: &'a Out,
}

pub trait RecordedOps: std::fmt::Debug {
    fn id(&self) -> NodeId;
    fn backward(&mut self);
    fn set_last_ops(&mut self);
    fn record(&self, tape: &mut Tape);
}
pub type RecordedOpsRef = Box<dyn RecordedOps>;
