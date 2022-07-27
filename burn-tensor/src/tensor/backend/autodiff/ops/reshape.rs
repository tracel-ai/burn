use crate::graph::node::{ForwardNode, ForwardNodeState};
use crate::graph::ops::ForwardUnaryRecordedOps;
use crate::tensor::backend::autodiff::ADKind;
use crate::tensor::{ops::*, Shape};
use crate::{
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    tensor::backend::autodiff::{ADCompatibleTensor, ADElement, ADTensor},
};

use std::sync::Arc;

#[derive(Debug)]
struct ADTensorOpsReshape<P, const D1: usize, const D2: usize> {
    shape: Shape<D1>,
    _kind: ADKind<P>,
}

impl<P: Default, const D1: usize, const D2: usize> ADTensorOpsReshape<P, D1, D2> {
    pub fn new(shape: Shape<D1>) -> Self {
        Self {
            shape,
            _kind: ADKind::new(),
        }
    }
}

impl<T1, T2, P, const D1: usize, const D2: usize> UnaryOps<T1, T2> for ADTensorOpsReshape<P, D1, D2>
where
    P: ADElement,
    T1: ADCompatibleTensor<P, D1> + TensorOpsReshape<P, D1, D2, T2>,
    T2: ADCompatibleTensor<P, D2> + TensorOpsReshape<P, D2, D1, T1>,
{
    fn partial(&self, state: &UnaryOpsNodeState<T1, T2>) -> T1 {
        state.output.grad().reshape(self.shape.clone())
    }
}

impl<P, const D1: usize, const D2: usize, T1, T2> TensorOpsReshape<P, D1, D2, ADTensor<P, D2, T2>>
    for ADTensor<P, D1, T1>
where
    P: ADElement,
    T1: ADCompatibleTensor<P, D1> + TensorOpsReshape<P, D1, D2, T2>,
    T2: ADCompatibleTensor<P, D2> + TensorOpsReshape<P, D2, D1, T1>,
{
    fn reshape(&self, shape: Shape<D2>) -> ADTensor<P, D2, T2> {
        let input = self.tensor();
        let out = TensorOpsReshape::reshape(&input, shape.clone());

        let state = ForwardNodeState::new(out);

        let ops = ADTensorOpsReshape::<P, D1, D2>::new(self.shape.clone());
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::ADTchTensor, Data};

    #[test]
    fn should_diff_mul() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 1> = Data::from([4.0, 7.0, 2.0, 3.0]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_2.reshape(Shape::new([2, 2]));
        let tensor_4 = &tensor_1.matmul(&tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([3.0, 3.0, 10.0, 10.0]));
    }
}
