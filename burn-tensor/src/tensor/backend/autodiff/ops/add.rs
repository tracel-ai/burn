use crate::graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState};
use crate::tensor::backend::autodiff::{ADCompatibleTensor, ADElement, ADTensor};
use crate::tensor::ops::*;
use crate::{execute_ops, register_ops};

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorAddOps,
    partial_left |state: &BinaryOpsNodeState<T, T, T>| {
        state.output.grad()
    },
    partial_right |state: &BinaryOpsNodeState<T, T, T>| {
        state.output.grad()
    },
);

register_ops!(
    ops UnaryOps<T, T>,
    name ADTensorAddScalarOps state P,
    partial |_state, state_recorded: &UnaryOpsNodeState<T, T>|  {
        state_recorded.output.grad()
    },
);

impl<T, P, const D: usize> TensorOpsAdd<P, D> for ADTensor<P, D, T>
where
    T: ADCompatibleTensor<P, D>,
    P: ADElement,
{
    fn add(&self, other: &Self) -> Self {
        let node = execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsAdd::add(&self.tensor(), &other.tensor()),
            ops ADTensorAddOps::new(),
        );
        self.from_existing(node)
    }

    fn add_scalar(&self, other: &P) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsAdd::add_scalar(&self.tensor(), &other),
            ops ADTensorAddScalarOps::new(other.clone()),
        );
        self.from_existing(node)
    }
}

impl<T, P, const D: usize> std::ops::Add<P> for ADTensor<P, D, T>
where
    T: ADCompatibleTensor<P, D> + 'static,
    P: ADElement + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}

impl<T, P, const D: usize> std::ops::Add<ADTensor<P, D, T>> for ADTensor<P, D, T>
where
    T: ADCompatibleTensor<P, D> + 'static,
    P: ADElement + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::ADTchTensor, Data};

    #[test]
    fn should_diff_add() {
        let data_1 = Data::from([2.0, 5.0]);
        let data_2 = Data::from([4.0, 1.0]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.clone() + tensor_2.clone();
        let grads = tensor_3.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(grad_2.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_3.into_data(), Data::from([6.0, 6.0]));
    }

    #[test]
    fn should_diff_add_scalar() {
        let data = Data::from([2.0, 10.0]);

        let tensor = ADTchTensor::from_data(data.clone());
        let tensor_out = tensor.clone() + 5.0;
        let grads = tensor_out.backward();

        let grad = grads.wrt(&tensor).unwrap();

        assert_eq!(grad.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_out.into_data(), Data::from([7.0, 15.0]));
    }

    #[test]
    fn test_add_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());
        let tensor_3 = ADTchTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.add(&tensor_2);
        let tensor_5 = tensor_4
            .add(&tensor_3)
            .add_scalar(&5.0)
            .add(&tensor_1)
            .add(&tensor_2);
        let tensor_6 = tensor_1.add(&tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[3.0, 3.0], [3.0, 3.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 2.0], [2.0, 2.0]]));
    }
}
