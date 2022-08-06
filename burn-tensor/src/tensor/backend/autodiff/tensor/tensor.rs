use crate::{
    execute_ops,
    graph::node::ForwardNodeRef,
    tensor::{backend::Backend, ops::TensorOpsUtilities, Data, Shape},
};

#[derive(Debug, Clone)]
pub struct ADTensor<const D: usize, B: Backend> {
    pub node: ForwardNodeRef<B::TensorPrimitive<D>>,
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
    pub fn from_tensor(tensor: B::TensorPrimitive<D>) -> Self {
        let node = execute_ops!(
            init tensor.clone()
        );

        let shape = tensor.shape().clone();
        Self { node, shape }
    }

    pub fn from_existing(&self, node: ForwardNodeRef<B::TensorPrimitive<D>>) -> Self {
        let shape = self.shape.clone();
        Self { node, shape }
    }
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn tensor(&self) -> B::TensorPrimitive<D> {
        self.node.state.value()
    }
}

#[cfg(test)]
pub mod helper {
    #[cfg(feature = "ndarray")]
    mod helper_impl {
        use crate::tensor::backend::autodiff::ADBackendNdArray;
        use crate::tensor::Tensor;

        pub type TestADTensor<E, const D: usize> = Tensor<D, ADBackendNdArray<E>>;
    }
    pub use helper_impl::*;

    #[cfg(feature = "tch")]
    #[cfg(not(feature = "ndarray"))]
    mod helper_impl {
        use crate::tensor::backend::autodiff::ADBackendTch;
        use crate::tensor::Tensor;

        pub type TestADTensor<E, const D: usize> = Tensor<D, ADBackendTch<E>>;
    }
    pub use helper_impl::*;
}
