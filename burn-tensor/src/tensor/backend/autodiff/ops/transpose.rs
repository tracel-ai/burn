use crate::graph::ops::{UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::backend::backend::Backend;
use crate::tensor::ops::*;
use crate::{execute_ops, register_ops};

register_ops!(
    ops UnaryOps,
    name ADTensorTransposeOps,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state.output.grad().transpose()
    },
);

impl<B: Backend, const D: usize> TensorOpsTranspose<B::Elem, D> for ADTensor<D, B> {
    fn transpose(&self) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsTranspose::transpose(&self.tensor()),
            ops ADTensorTransposeOps::<B, D>::new(),
        );
        self.from_existing(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_transpose() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2.transpose());
        let tensor_4 = tensor_3.transpose();
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[6.0, 10.0], [6.0, 10.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[3.0, 10.0], [3.0, 10.0]]));
    }
}
