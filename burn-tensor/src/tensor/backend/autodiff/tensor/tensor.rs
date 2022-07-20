use super::ADKind;
use crate::{
    node::{Node, NodeRef, NodeState, Ones, Zeros},
    ops::InitRecordedOps,
    FloatTensor,
};
use crate::{Shape, TensorBase};
use num_traits::Float;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct ADTensor<P, const D: usize, T> {
    pub node: NodeRef<T>,
    pub shape: Shape<D>,
    pub kind: ADKind<P>,
}

impl<T, P, const D: usize> TensorBase<P, D> for ADTensor<P, D, T>
where
    P: Float + Zeros<P> + Default + 'static,
    T: FloatTensor<P, D> + Clone + Zeros<T> + Ones<T> + 'static,
{
    fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    fn into_data(self) -> crate::Data<P, D> {
        self.tensor().into_data()
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T>
where
    P: Float + Zeros<P> + Default + 'static,
    T: FloatTensor<P, D> + Clone + Zeros<T> + Ones<T> + 'static,
{
    pub fn from_tensor(tensor: T) -> Self {
        let shape = tensor.shape().clone();
        let kind = ADKind::new();
        let state = NodeState::new_mut(tensor);

        let ops = InitRecordedOps::new(state.clone());
        let ops = Rc::new(ops);
        let node = Rc::new(Node::new(state, ops));

        Self { node, shape, kind }
    }

    pub fn from_existing(&self, node: NodeRef<T>) -> Self {
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self { node, shape, kind }
    }
}

impl<T: Clone + std::fmt::Debug, P, const D: usize> ADTensor<P, D, T> {
    pub fn tensor(&self) -> T {
        self.node.state.borrow().value()
    }
}

#[cfg(test)]
pub mod helper {
    use super::*;
    use crate::{
        backend::{autodiff::ADFloat, tch::TchTensor},
        Data,
    };

    pub type ADTchTensor<P, const D: usize> = ADTensor<P, D, TchTensor<P, D>>;

    impl<P: ADFloat + tch::kind::Element + Into<f64>, const D: usize> ADTchTensor<P, D> {
        pub fn from_data(data: Data<P, D>) -> Self {
            let tensor = TchTensor::from_data(data, tch::Device::Cpu);
            ADTensor::from_tensor(tensor)
        }
    }
}
