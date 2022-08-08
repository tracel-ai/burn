use crate::tensor::backend::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops BinaryOps,
    name ADTensorMulOps,
    partial_left |state: &BinaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        state.output.grad().mul(&state.right.value())
    },
    partial_right |state: &BinaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        state.output.grad().mul(&state.left.value())
    },
);

register_ops!(
    ops UnaryOps,
    name ADTensorMulScalarOps state B::Elem,
    partial |state, state_recorded: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        state_recorded.output.grad().mul_scalar(state)
    },
);

impl<B: Backend, const D: usize> TensorOpsMul<B::Elem, D> for ADTensor<D, B> {
    fn mul(&self, other: &Self) -> Self {
        let node = execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsMul::mul(&self.tensor(), &other.tensor()),
            ops ADTensorMulOps::<B, D>::new(),
        );
        self.from_existing(node)
    }

    fn mul_scalar(&self, other: &B::Elem) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsMul::mul_scalar(&self.tensor(), &other),
            ops ADTensorMulScalarOps::<B, D>::new(other.clone()),
        );
        self.from_existing(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mul() {
        let data_1 = Data::from([1.0, 7.0]);
        let data_2 = Data::from([4.0, 7.0]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.clone().mul(&tensor_2);
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), data_2);
        assert_eq!(grad_2.to_data(), data_1);
        assert_eq!(tensor_3.into_data(), Data::from([4.0, 49.0]));
    }

    #[test]
    fn should_diff_mul_scalar() {
        let data = Data::from([2.0, 5.0]);

        let tensor = TestADTensor::from_data(data.clone());
        let tensor_out = tensor.clone().mul_scalar(&4.0);

        let grads = tensor_out.backward();
        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(tensor_out.into_data(), Data::from([8.0, 20.0]));
        assert_eq!(grad.to_data(), Data::from([4.0, 4.0]));
    }

    #[test]
    fn test_mul_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_3 = TestADTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.mul(&tensor_2);
        let tensor_5 = tensor_4.mul(&tensor_3);
        let tensor_6 = tensor_1.mul(&tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(
            grad_1.to_data(),
            Data::from([[16.0, 196.0], [104.0, -36.0]])
        );
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 98.0], [338.0, 18.0]]));
    }
}
