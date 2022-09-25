use crate::graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::Backend;
use crate::{
    execute_ops, register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops BinaryOps,
    name ADTensorSubOps,
    partial_left |state: &BinaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        state.output.grad()
    },
    partial_right |state: &BinaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        state.output.grad().neg()
    },
);

register_ops!(
    ops UnaryOps,
    name ADTensorSubScalarOps state B::Elem,
    partial |_state, state_recorded: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state_recorded.output.grad()
    },
);

impl<B: Backend, const D: usize> TensorOpsSub<B::Elem, D> for ADTensor<D, B> {
    fn sub(&self, other: &Self) -> Self {
        execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsSub::sub(&self.tensor(), &other.tensor()),
            ops ADTensorSubOps::<B, D>::new(),
        )
    }

    fn sub_scalar(&self, other: &B::Elem) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsSub::sub_scalar(&self.tensor(), other),
            ops ADTensorSubScalarOps::<B, D>::new(*other),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_sub() {
        let data_1 = Data::from([2.0, 5.0]);
        let data_2 = Data::from([4.0, 1.0]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.sub(&tensor_2);
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(grad_2.to_data(), Data::from([-1.0, -1.0]));
        assert_eq!(tensor_3.into_data(), Data::from([-2.0, 4.0]));
    }

    #[test]
    fn should_diff_sub_scalar() {
        let data = Data::from([2.0, 10.0]);
        let tensor = TestADTensor::from_data(data);
        let tensor_out = tensor.sub_scalar(5.0);
        let grads = tensor_out.backward();

        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(grad.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_out.into_data(), Data::from([-3.0, 5.0]));
    }

    #[test]
    fn test_sub_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);
        let tensor_3 = TestADTensor::from_data(data_3);

        let tensor_4 = tensor_1.sub(&tensor_2);
        let tensor_5 = tensor_4.sub(&tensor_3).sub_scalar(5.0);
        let tensor_6 = tensor_1.sub(&tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[0.0, 0.0], [0.0, 0.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[1.0, 1.0], [1.0, 1.0]]));
    }
}
