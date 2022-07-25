use super::ADKind;
use crate::{
    backend::autodiff::{ADCompatibleTensor, ADElement},
    execute_ops,
    node::ForwardNodeRef,
};
use crate::{Shape, TensorBase};

#[derive(Debug, Clone)]
pub struct ADTensor<P, const D: usize, T> {
    pub node: ForwardNodeRef<T>,
    pub shape: Shape<D>,
    pub kind: ADKind<P>,
}

impl<T, P, const D: usize> TensorBase<P, D> for ADTensor<P, D, T>
where
    P: ADElement,
    T: ADCompatibleTensor<P, D>,
{
    fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    fn into_data(self) -> crate::Data<P, D> {
        self.tensor().into_data()
    }
    fn to_data(&self) -> crate::Data<P, D> {
        self.tensor().to_data()
    }
}

impl<T, P, const D: usize> ADTensor<P, D, T>
where
    P: ADElement,
    T: ADCompatibleTensor<P, D>,
{
    pub fn from_tensor(tensor: T) -> Self {
        let node = execute_ops!(
            init tensor.clone()
        );

        let shape = tensor.shape().clone();
        let kind = ADKind::new();
        Self { node, shape, kind }
    }

    pub fn from_existing(&self, node: ForwardNodeRef<T>) -> Self {
        let shape = self.shape.clone();
        let kind = self.kind.clone();

        Self { node, shape, kind }
    }
}

impl<T: Clone + std::fmt::Debug, P, const D: usize> ADTensor<P, D, T> {
    pub fn tensor(&self) -> T {
        self.node.state.value()
    }
}

#[cfg(test)]
pub mod helper {
    use super::*;
    use crate::{
        backend::{autodiff::ADElement, tch::TchTensor},
        Data,
    };

    pub type ADTchTensor<P, const D: usize> = ADTensor<P, D, TchTensor<P, D>>;

    impl<P: ADElement + tch::kind::Element + Into<f64>, const D: usize> ADTchTensor<P, D> {
        pub fn from_data(data: Data<P, D>) -> Self {
            let tensor = TchTensor::from_data(data, tch::Device::Cpu);
            ADTensor::from_tensor(tensor)
        }
    }
}
