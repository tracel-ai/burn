use crate::tensor::backend::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops UnaryOps,
    name ADTensorLogOps,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        let value = state.input.value();
        let value = value.ones().div(&value);
        state.output.grad().mul(&value)
    },
);

impl<B: Backend, const D: usize> TensorOpsLog<B::Elem, D> for ADTensor<D, B> {
    fn log(&self) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsLog::log(&self.tensor()),
            ops ADTensorLogOps::<B, D>::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_log() {
        let data_1 = Data::<f64, 2>::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::<f64, 2>::from([[6.0, 7.0], [9.0, 10.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2.log());
        let tensor_4 = tensor_3.matmul(&tensor_2);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[60.2652, 72.3130], [60.2652, 72.3130]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[22.8614, 24.5043], [24.5729, 26.8507]]), 3);
    }
}
