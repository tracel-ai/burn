use crate::tensor::ops::Zeros;
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
    pub fn value_ref(&self) -> &Out {
        &self.value
    }
}

#[derive(Debug, Clone)]
pub struct BackwardNodeState<Out> {
    pub value: Out,
    pub grad: RefCell<Out>,
}

impl<Out: Zeros<Out>> BackwardNodeState<Out> {
    pub fn new(value: Out) -> Self {
        let grad = value.zeros();
        let grad = RefCell::new(grad);

        Self { value, grad }
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
    pub fn grad(&self) -> Out {
        self.grad.borrow().clone()
    }

    pub fn update_grad(&self, grad: Out) {
        self.grad.swap(&RefCell::new(self.grad() + grad));
    }
}
