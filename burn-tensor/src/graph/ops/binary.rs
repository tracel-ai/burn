use super::{BinaryRecordedState, RecordedOps};
use crate::node::{Node, NodeId, NodeRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait BinaryOps<Lhs, Rhs, Out>: std::fmt::Debug {
    fn forward(&self, left: Lhs, right: Rhs) -> Out;
    fn partial_left(&self, state: &BinaryRecordedState<Lhs, Rhs, Out>) -> Lhs;
    fn partial_right(&self, state: &BinaryRecordedState<Lhs, Rhs, Out>) -> Rhs;
}

#[derive(Debug)]
pub struct BinaryOpsNode<Lhs, Rhs, Out> {
    pub id: NodeId,
    pub parent_left: NodeRef<Lhs>,
    pub parent_right: NodeRef<Rhs>,
    pub value: Out,
    pub grad: Option<Out>,
}

#[derive(new, Debug)]
pub struct BinaryRecordedOps<Lhs, Rhs, Out, Ops> {
    lhs: NodeRef<Lhs>,
    rhs: NodeRef<Rhs>,
    out: NodeRef<Out>,
    ops: Ops,
}

impl<Lhs, Rhs, Out> BinaryOpsNode<Lhs, Rhs, Out> {
    pub fn new(parent_left: NodeRef<Lhs>, parent_right: NodeRef<Rhs>, value: Out) -> Self {
        Self {
            id: NodeId::new(),
            parent_left,
            parent_right,
            value,
            grad: None,
        }
    }
}

impl<Lhs, Rhs, Out> Node<Out> for BinaryOpsNode<Lhs, Rhs, Out>
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
    Lhs: Clone + Zeros<Lhs> + Mul<Out, Output = Lhs>,
    Rhs: Clone + Zeros<Rhs> + Mul<Out, Output = Rhs>,
    Out: Clone + Zeros<Out> + Ones<Out> + 'static,
    Lhs: std::fmt::Debug,
    Rhs: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: BinaryOps<Lhs, Rhs, Out>,
{
    fn id(&self) -> NodeId {
        self.out.borrow().id()
    }

    fn backward(&mut self) {
        let left = self.lhs.borrow().value();
        let right = self.rhs.borrow().value();
        let output = self.out.borrow().value();
        let state = BinaryRecordedState::new(&left, &right, &output);

        let partial_left = self.ops.partial_left(&state);
        let partial_right: Rhs = self.ops.partial_right(&state);
        let grad_mine = self.out.borrow_mut().grad();

        self.lhs
            .borrow_mut()
            .update_grad(partial_left * grad_mine.clone());
        self.rhs.borrow_mut().update_grad(partial_right * grad_mine);
    }

    fn set_last_ops(&mut self) {
        let value = self.out.borrow().value();
        self.out.borrow_mut().update_grad(value.ones());
    }
}
