use crate::tensor::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
    ElementConversion,
};

register_ops!(
    ops UnaryOps,
    name ADTensorErfOps,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        let value = state.input.value();
        let exponent = value.powf(2.0.to_elem()).neg();
        let numerator = exponent.exp().mul_scalar(&2.0.to_elem());
        let denominator = std::f64::consts::PI.sqrt().to_elem();
        let value = numerator.div_scalar(&denominator);
        state.output.grad().mul(&value)
    },
);

impl<B: Backend, const D: usize> TensorOpsErf<B::Elem, D> for ADTensor<D, B> {
    fn erf(&self) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsErf::erf(&self.tensor()),
            ops ADTensorErfOps::<B, D>::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_erf() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0], [9.0, 10.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2.erf());
        let tensor_4 = tensor_3.matmul(&tensor_2);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[32.0, 32.0], [32.0, 32.0]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[8.0, 8.0], [8.0, 8.0]]), 3);
    }
}
