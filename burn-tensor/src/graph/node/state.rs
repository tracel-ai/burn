use crate::node::Zeros;
use std::{cell::RefCell, ops::Add};

#[derive(new, Debug)]
pub struct ForwardNodeState<Out> {
    value: Out,
}
impl<Out> ForwardNodeState<Out>
where
    Out: Clone,
{
    pub fn value(&self) -> Out {
        self.value.clone()
    }
}

#[derive(Debug)]
pub struct BackwardNodeState<Out> {
    pub value: Out,
    pub grad: Option<Out>,
}
pub type BackwardNodeStateRef<Out> = RefCell<BackwardNodeState<Out>>;

impl<Out> BackwardNodeState<Out> {
    fn new(value: Out) -> Self {
        Self { value, grad: None }
    }
    pub fn new_mut(value: Out) -> BackwardNodeStateRef<Out> {
        RefCell::new(Self::new(value))
    }
}
impl<Out> BackwardNodeState<Out>
where
    Out: Clone,
{
    pub fn value(&self) -> Out {
        self.value.clone()
    }
}

impl<Out> BackwardNodeState<Out>
where
    Out: Zeros<Out> + Clone + Add<Output = Out>,
    Out: std::fmt::Debug,
{
    pub fn grad(&mut self) -> Out {
        let grad_self = match &self.grad {
            Some(val) => val.clone(),
            None => self.value.zeros(),
        };
        self.grad = Some(grad_self.clone());
        grad_self
    }

    pub fn update_grad(&mut self, grad: Out) {
        self.grad = Some(self.grad() + grad);
    }
}
