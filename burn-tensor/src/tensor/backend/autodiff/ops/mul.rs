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
    use super::*;
    use crate::{
        backend::tch::TchTensor,
        tape::{Tape, TapeRef},
        Data, TensorBase,
    };
    use std::cell::RefCell;

    #[test]
    fn should_diff_mul() {
        let tape = TapeRef::new(RefCell::new(Tape::new()));
        let data_1 = Data::from([1.0]);
        let data_2 = Data::from([4.0]);

        let tensor_1 = TchTensor::from_data(data_1.clone(), tch::Device::Cpu);
        let tensor_2 = TchTensor::from_data(data_2.clone(), tch::Device::Cpu);

        let tensor_ad_1 = ADTensor::from_tensor(tensor_1, tape.clone());
        let tensor_ad_2 = ADTensor::from_tensor(tensor_2, tape.clone());

        let tensor_ad_3 = tensor_ad_1.mul(&tensor_ad_2);
        let data_ad_3 = tensor_ad_3.tensor().into_data();
        assert_eq!(data_ad_3, Data::from([4.0]));

        tensor_ad_3.backprob();
        let grad_1 = tensor_ad_1.grad();
        let grad_2 = tensor_ad_2.grad();

        assert_eq!(grad_1.into_data(), data_2);
        assert_eq!(grad_2.into_data(), data_1);
    }
}
