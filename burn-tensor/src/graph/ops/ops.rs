use crate::node::NodeId;

#[derive(new)]
pub struct BinaryRecordedState<'a, Lhs, Rhs, Out> {
    pub left: &'a Lhs,
    pub right: &'a Rhs,
    pub output: &'a Out,
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
}
pub type RecordedOpsRef = Box<dyn RecordedOps>;
