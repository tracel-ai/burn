use crate::tensor::{Element, Tensor};
use crate::{
    execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{backend::autodiff::ADTensor, ops::*},
};

register_ops!(
    ops UnaryOps<T, T>,
    name ADTensorNegOps,
    partial |state: &UnaryOpsNodeState<T, T>|{
        state.output.grad().neg()
    },
);

impl<T, P, const D: usize> TensorOpsNeg<P, D> for ADTensor<P, D, T>
where
    T: Tensor<P, D>,
    P: Element,
{
    fn neg(&self) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsNeg::neg(&self.tensor()),
            ops ADTensorNegOps::new(),
        );
        self.from_existing(node)
    }
}

impl<T, P, const D: usize> std::ops::Neg for ADTensor<P, D, T>
where
    T: Tensor<P, D> + 'static,
    P: Element + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn neg(self) -> Self::Output {
        TensorOpsNeg::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[3.0, 3.0], [10.0, 10.0]]));
    }
}
