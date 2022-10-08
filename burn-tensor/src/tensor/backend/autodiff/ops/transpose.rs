use crate::graph::ops::{UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::backend::Backend;
use crate::tensor::ops::*;
use crate::{execute_ops, register_ops};

register_ops!(
    ops UnaryOps,
    name ADTensorTransposeOps,
    partial |state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state.output.grad().transpose()
    },
);

#[derive(Debug)]
struct DimState {
    dim1: usize,
    dim2: usize,
}

register_ops!(
    ops UnaryOps,
    name ADTensorSwapDimOps state DimState,
    partial |dims: &DimState, state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>|{
        state.output.grad().swap_dims(dims.dim2, dims.dim1)
    },
);

impl<B: Backend, const D: usize> TensorOpsTranspose<B::Elem, D> for ADTensor<D, B> {
    fn transpose(&self) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsTranspose::transpose(&self.tensor()),
            ops ADTensorTransposeOps::<B, D>::new(),
        )
    }
    fn swap_dims(&self, dim1: usize, dim2: usize) -> Self {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsTranspose::swap_dims(&self.tensor(), dim1, dim2),
            ops ADTensorSwapDimOps::<B, D>::new(DimState { dim1, dim2 }),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_transpose() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, 7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2.transpose());
        let tensor_4 = tensor_3.transpose();
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[6.0, 10.0], [6.0, 10.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[3.0, 10.0], [3.0, 10.0]]));
    }

    #[test]
    fn should_diff_swap_dims() {
        let data_1 = Data::<f64, 3>::from([[[0.0, 1.0], [3.0, 4.0]], [[6.0, 7.0], [9.0, 10.0]]]);
        let data_2 = Data::<f64, 3>::from([[[1.0, 4.0], [2.0, 5.0]], [[7.0, 10.0], [8.0, 11.0]]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2.swap_dims(0, 2));
        let tensor_4 = tensor_3.matmul(&tensor_2.swap_dims(1, 2));
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(
            grad_1.to_data(),
            Data::from([[[66., 78.], [66., 78.]], [[270., 306.], [270., 306.]]])
        );
        assert_eq!(
            grad_2.to_data(),
            Data::from([[[22., 286.], [28., 316.]], [[172., 652.], [190., 694.]]])
        );
    }
}
