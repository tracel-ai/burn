use crate::graph::node::{ForwardNode, ForwardNodeState};
use crate::graph::ops::ForwardUnaryRecordedOps;
use crate::tensor::backend::autodiff::{ADBackend, ADKind};
use crate::tensor::{ops::*, Shape, Tensor, TensorType};
use crate::tensor::{Backend, Element};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::ADTensor,
};
use rand::distributions::Standard;
use std::sync::Arc;

#[derive(Debug)]
struct ADTensorOpsReshape<P, const D1: usize, const D2: usize, B> {
    shape: Shape<D1>,
    _b: B,
    _kind: ADKind<P>,
}

impl<P: Default, const D1: usize, const D2: usize, B: Backend> ADTensorOpsReshape<P, D1, D2, B> {
    pub fn new(shape: Shape<D1>) -> Self {
        Self {
            shape,
            _b: B::default(),
            _kind: ADKind::new(),
        }
    }
}

impl<P, const D1: usize, const D2: usize, B> UnaryOps<Tensor<D1, B>, Tensor<D2, B>>
    for ADTensorOpsReshape<P, D1, D2, B>
where
    B: Backend<E = P> + TensorType<D1, B> + TensorType<D2, B>,
    P: Element,
    Standard: rand::distributions::Distribution<P>,
{
    fn partial(&self, state: &UnaryOpsNodeState<Tensor<D1, B>, Tensor<D2, B>>) -> Tensor<D1, B> {
        state.output.grad().reshape(self.shape.clone())
    }
}

macro_rules! define_impl {
    ($b:ty) => {
        impl<E, const D1: usize> TensorOpsReshape<E, D1, ADBackend<E, $b>>
            for ADTensor<E, D1, Tensor<D1, $b>>
        where
            E: Element,
            Standard: rand::distributions::Distribution<E>,
        {
            fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, ADBackend<E, $b>> {
                let input = self.tensor();
                let out = TensorOpsReshape::reshape(&input, shape.clone());

                let state = ForwardNodeState::new(out);

                let ops = ADTensorOpsReshape::<E, D1, D2, $b>::new(self.shape.clone());
                let ops = Arc::new(ops);
                let ops = ForwardUnaryRecordedOps::new(self.node.clone(), ops);
                let ops = Arc::new(ops);

                let node = ForwardNode::from_unary(&self.node, state, ops);
                let node = Arc::new(node);

                let shape = shape.clone();
                let kind = self.kind.clone();

                ADTensor { node, shape, kind }
            }
        }
    };
}

crate::register_ad_backend!();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mul() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 1> = Data::from([4.0, 7.0, 2.0, 3.0]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_2.reshape(Shape::new([2, 2]));
        let tensor_4 = &tensor_1.matmul(&tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([3.0, 3.0, 10.0, 10.0]));
    }
}
