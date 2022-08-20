use crate::tensor::backend::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops BinaryOps,
    name ADTensorDivOps,
    partial_left |state: &BinaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        let value = state.right.value();
        let tmp = value.ones().div(&value);

        state.output.grad().mul(&tmp)
    },
    partial_right |state: &BinaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        let value_left = state.left.value();
        let value_right = state.right.value();
        let tmp = value_left.neg().div(&value_right.mul(&value_right));

        state.output.grad().mul(&tmp)
    },
);

register_ops!(
    ops UnaryOps,
    name ADTensorDivScalarOps state B::Elem,
    partial |state: &B::Elem, state_recorded: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>| {
        let value = state_recorded.input.value();
        let tmp = value.ones().div_scalar(&state);

        state_recorded.output.grad().mul(&tmp)
    },
);

impl<B: Backend, const D: usize> TensorOpsDiv<B::Elem, D> for ADTensor<D, B> {
    fn div(&self, other: &Self) -> Self {
        execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsDiv::div(&self.tensor(), &other.tensor()),
            ops ADTensorDivOps::<B, D>::new(),
        )
    }

    fn div_scalar(&self, other: &B::Elem) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsDiv::div_scalar(&self.tensor(), &other),
            ops ADTensorDivScalarOps::<B, D>::new(other.clone()),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_div() {
        let data_1 = Data::from([1.0, 7.0]);
        let data_2 = Data::from([4.0, 7.0]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.clone().div(&tensor_2);
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([0.25, 0.1429]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([-0.0625, -0.1429]), 3);
    }

    #[test]
    fn should_diff_div_scalar() {
        let data = Data::from([1.0, 7.0]);

        let tensor = TestADTensor::from_data(data.clone());
        let tensor_out = tensor.clone().div_scalar(&4.0);

        let grads = tensor_out.backward();
        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(grad.to_data(), Data::from([0.25, 0.25]));
    }

    #[test]
    fn test_div_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_3 = TestADTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.div(&tensor_2);
        let tensor_5 = tensor_4.div(&tensor_3);

        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[0.1250, 0.0714], [0.25, 0.1667]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[-0.0312, -0.0714], [-1.6250, 0.1667]]), 3);
    }
}
