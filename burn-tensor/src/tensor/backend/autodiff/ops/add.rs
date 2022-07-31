use crate::graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::backend::Backend;
use crate::tensor::ops::*;
use crate::{execute_ops, register_ops};

register_ops!(
    ops BinaryOps,
    name ADTensorAddOps,
    partial_left |state: &BinaryOpsNodeState<B::Tensor<D>, B::Tensor<D>, B::Tensor<D>>| {
        state.output.grad()
    },
    partial_right |state: &BinaryOpsNodeState<B::Tensor<D>, B::Tensor<D>, B::Tensor<D>>| {
        state.output.grad()
    },
);
register_ops!(
    ops UnaryOps,
    name ADTensorAddScalarOps state B::Elem,
    partial |_state, state_recorded: &UnaryOpsNodeState<B::Tensor<D>, B::Tensor<D>>|  {
        state_recorded.output.grad()
    },
);

impl<B: Backend, const D: usize> TensorOpsAdd<B::Elem, D> for ADTensor<D, B> {
    fn add(&self, other: &Self) -> Self {
        let node = execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsAdd::add(&self.tensor(), &other.tensor()),
            ops ADTensorAddOps::<B, D>::new(),
        );
        self.from_existing(node)
    }

    fn add_scalar(&self, other: &B::Elem) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsAdd::add_scalar(&self.tensor(), &other),
            ops ADTensorAddScalarOps::<B, D>::new(other.clone()),
        );
        self.from_existing(node)
    }
}

impl<B: Backend, const D: usize> std::ops::Add<ADTensor<D, B>> for ADTensor<D, B> {
    type Output = ADTensor<D, B>;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_add() {
        let data_1 = Data::from([2.0, 5.0]);
        let data_2 = Data::from([4.0, 1.0]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.clone() + tensor_2.clone();
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(grad_2.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_3.into_data(), Data::from([6.0, 6.0]));
    }

    #[test]
    fn should_diff_add_scalar() {
        let data = Data::from([2.0, 10.0]);

        let tensor = TestADTensor::from_data(data.clone());
        let tensor_out = tensor.clone().add_scalar(&5.0);
        let grads = tensor_out.backward();

        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(grad.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_out.into_data(), Data::from([7.0, 15.0]));
    }

    #[test]
    fn test_add_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());
        let tensor_3 = TestADTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.add(&tensor_2);
        let tensor_5 = tensor_4
            .add(&tensor_3)
            .add_scalar(&5.0)
            .add(&tensor_1)
            .add(&tensor_2);
        let tensor_6 = tensor_1.add(&tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[3.0, 3.0], [3.0, 3.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 2.0], [2.0, 2.0]]));
    }
}
