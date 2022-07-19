use super::{BinaryRecordedState, RecordedOps, RecordedOpsRef};
use crate::node::{NodeId, NodeRef, NodeState, NodeStateRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait BinaryOps<Lhs, Rhs, Out>: std::fmt::Debug {
    fn partial_left(&self, state: &BinaryRecordedState<Lhs, Rhs, Out>) -> Lhs;
    fn partial_right(&self, state: &BinaryRecordedState<Lhs, Rhs, Out>) -> Rhs;
}

#[derive(Debug, Clone)]
pub struct BinaryOpsNode<Lhs, Rhs, Out> {
    pub id: NodeId,
    pub parent_left: NodeStateRef<Lhs>,
    pub parent_right: NodeStateRef<Rhs>,
    pub value: Out,
    pub grad: Option<Out>,
}

#[derive(new, Debug, Clone)]
pub struct BinaryRecordedOps<Lhs, Rhs, Out, Ops> {
    lhs: NodeRef<Lhs>,
    rhs: NodeRef<Rhs>,
    out: NodeStateRef<Out>,
    ops: Ops,
}

impl<Lhs, Rhs, Out> BinaryOpsNode<Lhs, Rhs, Out> {
    pub fn new(
        parent_left: NodeStateRef<Lhs>,
        parent_right: NodeStateRef<Rhs>,
        value: Out,
    ) -> Self {
        Self {
            id: NodeId::new(),
            parent_left,
            parent_right,
            value,
            grad: None,
        }
    }
}

impl<Lhs, Rhs, Out> NodeState<Out> for BinaryOpsNode<Lhs, Rhs, Out>
where
    Out: Zeros<Out> + Clone + Mul<Output = Out> + Add<Output = Out>,
    Lhs: std::fmt::Debug,
    Rhs: std::fmt::Debug,
    Out: std::fmt::Debug,
{
    fn id(&self) -> NodeId {
        self.id.clone()
    }
    fn value(&self) -> Out {
        self.value.clone()
    }

    fn grad(&mut self) -> Out {
        let grad_self = match &self.grad {
            Some(val) => val.clone(),
            None => self.value.zeros(),
        };
        self.grad = Some(grad_self.clone());
        grad_self
    }

    fn update_grad(&mut self, grad: Out) {
        self.grad = Some(self.grad() + grad);
    }
}

impl<Lhs, Rhs, Out, Ops> RecordedOps for BinaryRecordedOps<Lhs, Rhs, Out, Ops>
where
    Lhs: Clone + Zeros<Lhs> + Mul<Out, Output = Lhs> + 'static,
    Rhs: Clone + Zeros<Rhs> + Mul<Out, Output = Rhs> + 'static,
    Out: Clone + Zeros<Out> + Ones<Out> + 'static,
    Lhs: std::fmt::Debug,
    Rhs: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: BinaryOps<Lhs, Rhs, Out> + 'static + Clone,
{
    fn id(&self) -> NodeId {
        self.out.borrow().id()
    }

    fn backward(&self) {
        let state = BinaryRecordedState::new(&self.lhs.state, &self.rhs.state, &self.out);

        let partial_left = self.ops.partial_left(&state);
        let partial_right: Rhs = self.ops.partial_right(&state);

        let grad_mine = self.out.borrow_mut().grad();

        self.lhs
            .state
            .borrow_mut()
            .update_grad(partial_left * grad_mine.clone());
        self.rhs
            .state
            .borrow_mut()
            .update_grad(partial_right * grad_mine);
    }

    fn set_last_ops(&self) {
        let value = self.out.borrow().value();
        self.out.borrow_mut().update_grad(value.ones());
    }

    fn parents_ops(&self) -> Vec<RecordedOpsRef> {
        vec![self.lhs.ops.clone(), self.rhs.ops.clone()]
    }
}
