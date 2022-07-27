use crate::graph::node::{ForwardNode, ForwardNodeState};
use crate::graph::ops::{
    BinaryOps, BinaryOpsNodeState, ForwardBinaryRecordedOps, ForwardUnaryRecordedOps, UnaryOps,
    UnaryOpsNodeState,
};
use crate::tensor::backend::autodiff::{ADKind, ADTensor};
use crate::tensor::ops::*;
use crate::tensor::{Element, Tensor};
use std::{ops::Range, sync::Arc};

#[derive(Debug)]
struct ADTensorOpsIndex<P, const D1: usize, const D2: usize> {
    indexes: [Range<usize>; D2],
    _kind: ADKind<P>,
}

impl<P: Default, const D1: usize, const D2: usize> ADTensorOpsIndex<P, D1, D2> {
    pub fn new(indexes: [Range<usize>; D2]) -> Self {
        Self {
            indexes,
            _kind: ADKind::new(),
        }
    }
}

impl<T1, P, const D1: usize, const D2: usize> UnaryOps<T1, T1> for ADTensorOpsIndex<P, D1, D2>
where
    P: Element,
    T1: Tensor<P, D1> + TensorOpsIndex<P, D1, D2>,
{
    fn partial(&self, state: &UnaryOpsNodeState<T1, T1>) -> T1 {
        state
            .input
            .value()
            .zeros()
            .index_assign(self.indexes.clone(), &state.output.grad())
    }
}

#[derive(Debug)]
struct ADTensorOpsIndexAssign<P, const D1: usize, const D2: usize> {
    indexes: [Range<usize>; D2],
    _kind: ADKind<P>,
}

impl<P: Default, const D1: usize, const D2: usize> ADTensorOpsIndexAssign<P, D1, D2> {
    pub fn new(indexes: [Range<usize>; D2]) -> Self {
        Self {
            indexes,
            _kind: ADKind::new(),
        }
    }
}

impl<T, P, const D1: usize, const D2: usize> BinaryOps<T, T, T>
    for ADTensorOpsIndexAssign<P, D1, D2>
where
    P: Element,
    T: Tensor<P, D1> + TensorOpsIndex<P, D1, D2>,
{
    fn partial_left(&self, state: &BinaryOpsNodeState<T, T, T>) -> T {
        state
            .output
            .grad()
            .index_assign(self.indexes.clone(), &state.right.value().zeros())
    }

    fn partial_right(&self, state: &BinaryOpsNodeState<T, T, T>) -> T {
        state.output.grad().index(self.indexes.clone())
    }
}

impl<P, const D1: usize, const D2: usize, T> TensorOpsIndex<P, D1, D2> for ADTensor<P, D1, T>
where
    P: Element,
    T: Tensor<P, D1> + TensorOpsIndex<P, D1, D2>,
{
    fn index(&self, indexes: [Range<usize>; D2]) -> Self {
        let input = self.tensor();
        let out = TensorOpsIndex::index(&input, indexes.clone());
        let shape = out.shape().clone();

        let state = ForwardNodeState::new(out);

        let ops = ADTensorOpsIndex::<P, D1, D2>::new(indexes);
        let ops = Arc::new(ops);
        let ops = ForwardUnaryRecordedOps::new(self.node.clone(), ops);
        let ops = Arc::new(ops);

        let node = ForwardNode::from_unary(&self.node, state, ops);
        let node = Arc::new(node);

        let kind = self.kind.clone();

        Self { node, shape, kind }
    }
    fn index_assign(&self, indexes: [Range<usize>; D2], values: &Self) -> Self {
        let input = self.tensor();
        let out = TensorOpsIndex::index_assign(&input, indexes.clone(), &values.tensor());
        let shape = out.shape().clone();

        let state = ForwardNodeState::new(out);

        let ops = ADTensorOpsIndexAssign::<P, D1, D2>::new(indexes);
        let ops = Arc::new(ops);
        let ops = ForwardBinaryRecordedOps::new(self.node.clone(), values.node.clone(), ops);
        let ops = Arc::new(ops);

        let node = ForwardNode::from_binary(&self.node, &values.node, state, ops);
        let node = Arc::new(node);

        let kind = self.kind.clone();

        Self { node, shape, kind }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::ADTchTensor, Data};

    #[test]
    fn should_diff_matmul_with_index() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0, 100.0], [2.0, 3.0, 15.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_2.index([0..2, 0..2]);
        let tensor_4 = &tensor_1.matmul(&tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(
            grad_2.to_data(),
            Data::from([[3.0, 3.0, 0.0], [10.0, 10.0, 0.0]])
        );
    }

    #[test]
    fn should_diff_matmul_with_index_assign() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_assigned: Data<f64, 2> = Data::from([[9.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());
        let tensor_assigned = ADTchTensor::from_data(data_assigned.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.index_assign([0..1, 0..1], &tensor_assigned);
        let tensor_5 = &tensor_4.matmul(&tensor_1);

        let grads = tensor_5.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[58.0, 38.0], [118.0, 82.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[16.0, 15.0], [24.0, 50.0]]));
    }

    #[test]
    fn should_diff_matmul_with_index_assign_complex() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[9.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());
        let tensor_3 = ADTchTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.matmul(&tensor_2);
        let tensor_5 = tensor_2.index([0..1, 0..1]);
        let tensor_6 = tensor_5.mul(&tensor_3);
        let tensor_7 = tensor_4.index_assign([0..1, 0..1], &tensor_6);
        let tensor_8 = &tensor_7.matmul(&tensor_1);

        let grads = tensor_8.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();
        let grad_3 = grads.wrt(&tensor_3).unwrap();

        assert_eq!(grad_3.to_data(), Data::from([[32.0]]));
        assert_eq!(grad_1.to_data(), Data::from([[85.0, 65.0], [118.0, 82.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[88.0, 15.0], [24.0, 50.0]]));
    }
}
