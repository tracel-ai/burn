use crate::execute_ops;
use crate::graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::backend::Backend;
use crate::tensor::ops::*;
use std::ops::Range;

#[derive(Debug)]
struct ADTensorOpsIndex<B: Backend, const D1: usize, const D2: usize> {
    indexes: [Range<usize>; D2],
    _b: B,
}

#[derive(Debug)]
struct ADTensorOpsIndexAssign<B: Backend, const D1: usize, const D2: usize> {
    indexes: [Range<usize>; D2],
    _b: B,
}

impl<B: Backend, const D1: usize, const D2: usize> ADTensorOpsIndex<B, D1, D2> {
    pub fn new(indexes: [Range<usize>; D2]) -> Self {
        Self {
            indexes,
            _b: B::default(),
        }
    }
}

impl<B: Backend, const D1: usize, const D2: usize> ADTensorOpsIndexAssign<B, D1, D2> {
    pub fn new(indexes: [Range<usize>; D2]) -> Self {
        Self {
            indexes,
            _b: B::default(),
        }
    }
}

impl<B: Backend, const D1: usize, const D2: usize>
    UnaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D1>> for ADTensorOpsIndex<B, D1, D2>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D1>, B::TensorPrimitive<D1>>,
    ) -> B::TensorPrimitive<D1> {
        state
            .input
            .value()
            .zeros()
            .index_assign(self.indexes.clone(), &state.output.grad())
    }
}

impl<B: Backend, const D1: usize, const D2: usize>
    BinaryOps<B::TensorPrimitive<D1>, B::TensorPrimitive<D1>, B::TensorPrimitive<D1>>
    for ADTensorOpsIndexAssign<B, D1, D2>
{
    fn partial_left(
        &self,
        state: &BinaryOpsNodeState<
            B::TensorPrimitive<D1>,
            B::TensorPrimitive<D1>,
            B::TensorPrimitive<D1>,
        >,
    ) -> B::TensorPrimitive<D1> {
        state
            .output
            .grad()
            .index_assign(self.indexes.clone(), &state.right.value().zeros())
    }

    fn partial_right(
        &self,
        state: &BinaryOpsNodeState<
            B::TensorPrimitive<D1>,
            B::TensorPrimitive<D1>,
            B::TensorPrimitive<D1>,
        >,
    ) -> B::TensorPrimitive<D1> {
        state.output.grad().index(self.indexes.clone())
    }
}

impl<B: Backend, const D1: usize> TensorOpsIndex<B::Elem, D1> for ADTensor<D1, B> {
    fn index<const D2: usize>(&self, indexes: [Range<usize>; D2]) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsIndex::index(&self.tensor(), indexes.clone()),
            ops ADTensorOpsIndex::<B, D1, D2>::new(indexes),
        )
    }
    fn index_assign<const D2: usize>(&self, indexes: [Range<usize>; D2], values: &Self) -> Self {
        execute_ops!(
            lhs self.node.clone(),
            rhs values.node.clone(),
            out TensorOpsIndex::index_assign::<D2>(&self.tensor(), indexes.clone(), &values.tensor()),
            ops ADTensorOpsIndexAssign::<B, D1, D2>::new(indexes),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_matmul_with_index() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0, 100.0], [2.0, 3.0, 15.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_2.index([0..2, 0..2]);
        let tensor_4 = &tensor_1.matmul(&tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

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

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_assigned = TestADTensor::from_data(data_assigned.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.index_assign([0..1, 0..1], &tensor_assigned);
        let tensor_5 = &tensor_4.matmul(&tensor_1);

        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[58.0, 38.0], [118.0, 82.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[16.0, 15.0], [24.0, 50.0]]));
    }

    #[test]
    fn should_diff_matmul_with_index_assign_complex() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[9.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_3 = TestADTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.matmul(&tensor_2);
        let tensor_5 = tensor_2.index([0..1, 0..1]);
        let tensor_6 = tensor_5.mul(&tensor_3);
        let tensor_7 = tensor_4.index_assign([0..1, 0..1], &tensor_6);
        let tensor_8 = &tensor_7.matmul(&tensor_1);

        let grads = tensor_8.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();
        let grad_3 = tensor_3.grad(&grads).unwrap();

        assert_eq!(grad_3.to_data(), Data::from([[32.0]]));
        assert_eq!(grad_1.to_data(), Data::from([[85.0, 65.0], [118.0, 82.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[88.0, 15.0], [24.0, 50.0]]));
    }
}
