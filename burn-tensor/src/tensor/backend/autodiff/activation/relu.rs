use crate::tensor::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    ops::activation::*,
    ops::*,
    register_ops,
    tensor::backend::autodiff::ADTensor,
};

register_ops!(
    ops UnaryOps,
    name ADReLU,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        let zero = B::Elem::zeros(&B::Elem::default());
        let mask = state.output.value().lower_equal_scalar(&zero);

        B::mask_fill(&state.output.grad(), &mask, zero)

    },
);

impl<B: Backend, const D: usize> ReLU<B::Elem, D> for ADTensor<D, B> {
    fn relu(&self) -> Self {
        execute_ops!(
            input self.node.clone(),
            out self.tensor_ref().relu(),
            ops ADReLU::<B, D>::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_relu() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_3.relu();
        let tensor_5 = tensor_4.matmul(&tensor_2);
        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[-47.0, 9.0], [-35.0, 15.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[15.0, 13.0], [-2.0, 39.0]]));
    }
}
