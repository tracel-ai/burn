use crate::{
    backend::autodiff::{ADFloat, ADFloatTensor, ADTensor},
    define_ops, execute_ops,
    ops::{
        BinaryOps, BinaryRecordedOps, BinaryRecordedState, SingleOps, SingleRecordedOps,
        SingleRecordedState,
    },
    register_ops, TensorOpsMul,
};
use num_traits::Float;

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorMulOps,
    forward |left, right| left * right,
    partial_left |state: &BinaryRecordedState<T, T, T>| state.right.clone(),
    partial_right |state: &BinaryRecordedState<T, T, T>| state.left.clone(),
);

register_ops!(
    ops SingleOps<T, T>,
    name ADTensorMulScalarOps state P,
    forward |state, input|  input * state,
    partial |state, state_recorded: &SingleRecordedState<T, T>|  state_recorded.input.ones() * state,
);

impl<T, P, const D: usize> TensorOpsMul<P, D> for ADTensor<P, D, T>
where
    T: ADFloatTensor<P, D>,
    P: ADFloat,
{
    fn mul(&self, other: &Self) -> Self {
        let node = execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsMul::mul(&self.tensor(), &other.tensor()),
            tape self.tape.clone(),
            ops ADTensorMulOps::new(),
        );
        self.from_existing(node)
    }

    fn mul_scalar(&self, other: &P) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsMul::mul_scalar(&self.tensor(), &other),
            tape self.tape.clone(),
            ops ADTensorMulScalarOps::new(other.clone()),
        );
        self.from_existing(node)
    }
}

impl<T, P, const D: usize> std::ops::Mul<P> for ADTensor<P, D, T>
where
    T: ADFloatTensor<P, D> + 'static,
    P: ADFloat + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}

impl<T, P, const D: usize> std::ops::Mul<ADTensor<P, D, T>> for ADTensor<P, D, T>
where
    T: ADFloatTensor<P, D> + 'static,
    P: ADFloat + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{backend::autodiff::helper::ADTchTensor, tape::Tape, Data, TensorBase};

    #[test]
    fn should_diff_mul() {
        let tape = Tape::new_ref();
        let data_1 = Data::from([1.0]);
        let data_2 = Data::from([4.0]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone(), tape.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone(), tape.clone());

        let tensor_3 = tensor_1.clone() * tensor_2.clone();
        tensor_3.backprob();

        let grad_1 = tensor_1.grad();
        let grad_2 = tensor_2.grad();

        assert_eq!(grad_1.into_data(), data_2);
        assert_eq!(grad_2.into_data(), data_1);
        assert_eq!(tensor_3.into_data(), Data::from([4.0]));
    }

    #[test]
    fn should_diff_mul_scalar() {
        let tape = Tape::new_ref();
        let data = Data::from([2.0]);

        let tensor = ADTchTensor::from_data(data.clone(), tape.clone());
        let tensor_out = tensor.clone() * 4.0;
        tensor_out.backprob();

        let grad = tensor.grad();
        assert_eq!(tensor_out.into_data(), Data::from([8.0]));
        assert_eq!(grad.into_data(), Data::from([4.0]));
    }
}
