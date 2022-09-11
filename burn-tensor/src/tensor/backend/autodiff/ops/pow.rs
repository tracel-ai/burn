use crate::tensor::backend::backend::Backend;
use crate::ElementConversion;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops UnaryOps,
    name ADTensorPowOps state f32,
    partial |
        value: &f32,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
    | {
        let value = state.input
            .value()
            .powf(value - 1.0)
            .mul_scalar(&value.clone().to_elem());
        state.output.grad().mul(&value)
    },
);

impl<B: Backend, const D: usize> TensorOpsPow<B::Elem, D> for ADTensor<D, B> {
    fn powf(&self, value: f32) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsPow::powf(&self.tensor(), value),
            ops ADTensorPowOps::<B, D>::new(value.clone()),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_powf() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0], [9.0, 10.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2.powf(0.4));
        let tensor_4 = tensor_3.matmul(&tensor_2);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[68.0, 79.0328], [68.0, 79.0328]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[23.5081, 25.2779], [26.0502, 28.6383]]), 3);
    }
}
