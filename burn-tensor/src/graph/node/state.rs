use crate::tensor::ops::Zeros;
use std::ops::Add;
use std::sync::RwLock;

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
    value: Out,
    grad: RwLock<Out>,
}

impl<Out: Zeros<Out>> BackwardNodeState<Out> {
    pub fn new(value: Out) -> Self {
        let grad = value.zeros();
        let grad = RwLock::new(grad);

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
        self.grad.read().unwrap().clone()
    }

    pub fn update_grad(&self, grad: Out) {
        let mut grad_state = self.grad.write().unwrap();
        *grad_state = grad_state.clone() + grad;
    }
}
