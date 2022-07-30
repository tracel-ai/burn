use super::ADKind;
use crate::{
    execute_ops,
    graph::node::ForwardNodeRef,
    tensor::{backend::Backend, ops::TensorOpsUtilities, Data, Shape},
};

#[derive(Debug, Clone)]
pub struct ADTensor<const D: usize, B: Backend> {
    pub node: ForwardNodeRef<B::Tensor<D>>,
    pub shape: Shape<D>,
}

impl<B: Backend, const D: usize> TensorOpsUtilities<B::Elem, D> for ADTensor<D, B> {
    fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    fn into_data(self) -> Data<B::Elem, D> {
        self.tensor().into_data()
    }
    fn to_data(&self) -> Data<B::Elem, D> {
        self.tensor().to_data()
    }
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn from_tensor(tensor: B::Tensor<D>) -> Self {
        let node = execute_ops!(
            init tensor.clone()
        );

        let shape = tensor.shape().clone();
        Self { node, shape }
    }

    pub fn from_existing(&self, node: ForwardNodeRef<B::Tensor<D>>) -> Self {
        let shape = self.shape.clone();
        Self { node, shape }
    }
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn tensor(&self) -> B::Tensor<D> {
        self.node.state.value()
    }
}

#[cfg(test)]
pub mod helper {
    use super::*;

    #[cfg(feature = "ndarray")]
    mod helper_impl {
        use super::*;
        use crate::tensor::Tensor;
        use crate::tensor::backend::autodiff::ADBackendNdArray;
        use crate::tensor::Element;
        use crate::tensor::{backend::ndarray::NdArrayTensor, Data};
        use rand::distributions::Standard;

        pub type TestADTensor<E, const D: usize> = Tensor<D, ADBackendNdArray<E>>;

        impl<E: Element, const D: usize> TestADTensor<E, D>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            pub fn from_data(data: Data<E, D>) -> Self {
                let tensor = NdArrayTensor::from_data(data);
                let tensor = ADTensor::from_tensor(tensor);
                Tensor::new(tensor)
            }
        }
    }
    pub use helper_impl::*;

    #[cfg(feature = "tch")]
    #[cfg(not(feature = "ndarray"))]
    mod helper_impl {
        use super::*;
        use crate::tensor::backend::tch::TchTensor;

        pub type TestADTensor<P, const D: usize> = ADTensor<P, D, TchTensor<P, D>>;
        impl<P: Element + tch::kind::Element + Into<f64>, const D: usize> TestADTensor<P, D> {
            pub fn from_data(data: Data<P, D>) -> Self {
                let tensor = TchTensor::from_data(data, tch::Device::Cpu);
                ADTensor::from_tensor(tensor)
            }
        }
    }
    pub use helper_impl::*;
}
