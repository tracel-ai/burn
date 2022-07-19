use crate::node::{NodeId, Zeros};
use std::ops::{Add, Mul};

#[derive(Debug)]
pub struct NodeStateImpl<Out> {
    pub id: NodeId,
    pub value: Out,
    pub grad: Option<Out>,
}

impl<Out> NodeStateImpl<Out> {
    pub fn new(value: Out) -> Self {
        Self {
            id: NodeId::new(),
            value,
            grad: None,
        }
    }
}
impl<Out> NodeStateImpl<Out>
where
    Out: std::fmt::Debug + Clone,
{
    pub fn id(&self) -> NodeId {
        self.id.clone()
    }
    pub fn value(&self) -> Out {
        self.value.clone()
    }
}

impl<Out> NodeStateImpl<Out>
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
