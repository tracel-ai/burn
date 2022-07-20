use crate::{
    backend::autodiff::{ADFloat, ADFloatTensor, ADTensor},
    define_ops, execute_ops,
    ops::{
        BinaryOps, BinaryOpsNodeState, BinaryRecordedOps, UnaryOps, UnaryOpsNodeState,
        UnaryRecordedOps,
    },
    register_ops, TensorOpsAdd,
};
use num_traits::Float;

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorAddOps,
    partial_left |state: &BinaryOpsNodeState<T, T, T>| state.left.borrow().value().ones(),
    partial_right |state: &BinaryOpsNodeState<T, T, T>| state.right.borrow().value().ones(),
);

register_ops!(
    ops UnaryOps<T, T>,
    name ADTensorAddScalarOps state P,
    partial |_state, state_recorded: &UnaryOpsNodeState<T, T>|  state_recorded.input.ones(),
);

impl<T, P, const D: usize> TensorOpsAdd<P, D> for ADTensor<P, D, T>
where
    T: ADFloatTensor<P, D>,
    P: ADFloat,
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
    T: ADFloatTensor<P, D> + 'static,
    P: ADFloat + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn add(self, rhs: P) -> Self::Output {
        TensorOpsAdd::add_scalar(&self, &rhs)
    }
}

impl<T, P, const D: usize> std::ops::Add<ADTensor<P, D, T>> for ADTensor<P, D, T>
where
    T: ADFloatTensor<P, D> + 'static,
    P: ADFloat + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOpsAdd::add(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::autodiff::helper::ADTchTensor, Data, TensorBase};

    #[test]
    fn should_diff_add() {
        let data_1 = Data::from([2.0, 5.0]);
        let data_2 = Data::from([4.0, 1.0]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.clone() + tensor_2.clone();
        tensor_3.backward();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(grad_1.into_data(), Data::from([1.0, 1.0]));
        assert_eq!(grad_2.into_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_3.into_data(), Data::from([6.0, 6.0]));
    }

    #[test]
    fn should_diff_add_scalar() {
        let data = Data::from([2.0, 10.0]);

        let tensor = ADTchTensor::from_data(data.clone());
        let tensor_out = tensor.clone() + 5.0;
        tensor_out.backward();

        let grad = tensor.grad();
        assert_eq!(grad.into_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_out.into_data(), Data::from([7.0, 15.0]));
    }
}
