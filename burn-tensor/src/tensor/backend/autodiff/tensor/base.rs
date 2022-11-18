use crate::{
    execute_ops,
    graph::node::ForwardNodeRef,
    tensor::{backend::Backend, Shape},
};

#[derive(Debug, Clone)]
pub struct ADTensor<const D: usize, B: Backend> {
    pub node: ForwardNodeRef<B::TensorPrimitive<D>>,
    pub shape: Shape<D>,
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn from_tensor(tensor: B::TensorPrimitive<D>) -> Self {
        let node = execute_ops!(
            init tensor.clone()
        );

        let shape = *B::shape(&tensor);
        Self { node, shape }
    }
}

impl<B: Backend, const D: usize> ADTensor<D, B> {
    pub fn tensor(&self) -> B::TensorPrimitive<D> {
        self.node.state.value()
    }

    pub fn tensor_ref(&self) -> &B::TensorPrimitive<D> {
        self.node.state.value_ref()
    }
}
