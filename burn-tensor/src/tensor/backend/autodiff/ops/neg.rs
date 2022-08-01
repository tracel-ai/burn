use crate::tensor::backend::backend::Backend;
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops UnaryOps,
    name ADTensorNegOps,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state.output.grad().neg()
    },
);

impl<B: Backend, P, const D: usize> TensorOpsNeg<P, D> for ADTensor<D, B> {
    fn neg(&self) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsNeg::neg(&self.tensor()),
            ops ADTensorNegOps::<B, D>::new(),
        );
        self.from_existing(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_neg() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2.neg());
        let tensor_4 = tensor_3.neg();
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[3.0, 3.0], [10.0, 10.0]]));
    }
}
