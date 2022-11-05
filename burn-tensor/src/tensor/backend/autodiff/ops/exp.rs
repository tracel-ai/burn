use crate::tensor::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops UnaryOps,
    name ADTensorExpOps,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        B::mul(&state.output.grad(), &state.output.value())
    },
);

impl<B: Backend, const D: usize> TensorOpsExp<B::Elem, D> for ADTensor<D, B> {
    fn exp(&self) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsExp::exp(&self.tensor()),
            ops ADTensorExpOps::<B, D>::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_exp() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2.exp());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[54.5991, 27.4746], [54.5991, 27.4746]]), 3);
        grad_2.to_data().assert_approx_eq(
            &Data::from([[-5.4598e+01, -9.1188e-04], [2.9556e+01, 8.0342e+01]]),
            3,
        );
    }
}
