use super::{RecordedOps, SingleRecordedState};
use crate::node::{NodeId, NodeRef, NodeState, NodeStateRef, Ones, Zeros};
use std::ops::{Add, Mul};

pub trait SingleOps<In, Out>: std::fmt::Debug {
    fn partial(&self, state: &SingleRecordedState<In, Out>) -> In;
}

#[derive(Debug)]
pub struct SingleOpsNode<In, Out> {
    pub id: NodeId,
    pub parent: NodeStateRef<In>,
    pub value: Out,
    pub grad: Option<Out>,
}

#[derive(new, Debug, Clone)]
pub struct SingleRecordedOps<In, Out, Ops> {
    input: NodeRef<In>,
    out: NodeStateRef<Out>,
    ops: Ops,
}

impl<In, Out> SingleOpsNode<In, Out> {
    pub fn new(parent: NodeStateRef<In>, value: Out) -> Self {
        Self {
            id: NodeId::new(),
            parent,
            value,
            grad: None,
        }
    }
}

impl<In, Out> NodeState<Out> for SingleOpsNode<In, Out>
where
    Out: Zeros<Out> + Clone + Mul<Output = Out> + Add<Output = Out>,
    In: std::fmt::Debug,
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

impl<In, Out, Ops> RecordedOps for SingleRecordedOps<In, Out, Ops>
where
    In: Clone + Zeros<In> + Mul<Out, Output = In> + 'static,
    Out: Clone + Zeros<Out> + Ones<Out> + 'static,
    In: std::fmt::Debug,
    Out: std::fmt::Debug,
    Ops: SingleOps<In, Out> + 'static + Clone,
{
    fn id(&self) -> NodeId {
        self.out.borrow().id()
    }

    fn backward(&mut self) {
        let input = self.input.state.borrow().value();
        let output = self.out.borrow().value();
        let state = SingleRecordedState::new(&input, &output);

        let partial = self.ops.partial(&state);
        let grad_mine = self.out.borrow_mut().grad();

        self.input
            .state
            .borrow_mut()
            .update_grad(partial * grad_mine.clone());
    }

    fn set_last_ops(&mut self) {
        let value = self.out.borrow().value();
        self.out.borrow_mut().update_grad(value.ones());
    }

    fn record(&self, tape: &mut crate::tape::Tape) {
        tape.add(Box::new(self.clone()));
        self.input.record(tape);
    }
}
