use crate::node::Zeros;
use std::{cell::RefCell, ops::Add};

#[derive(Debug)]
pub struct NodeState<Out> {
    pub value: Out,
    pub grad: Option<Out>,
}
pub type NodeStateRef<Out> = RefCell<NodeState<Out>>;

impl<Out> NodeState<Out> {
    pub fn new(value: Out) -> Self {
        Self { value, grad: None }
    }
    pub fn new_mut(value: Out) -> NodeStateRef<Out> {
        RefCell::new(Self::new(value))
    }
}
impl<Out> NodeState<Out>
where
    Out: Clone,
{
    pub fn value(&self) -> Out {
        self.value.clone()
    }
}

impl<Out> NodeState<Out>
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
