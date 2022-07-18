use crate::{
    backend::autodiff::{ADFloat, ADFloatTensor, ADTensor},
    define_ops, execute_ops,
    ops::{
        BinaryOps, BinaryRecordedOps, BinaryRecordedState, SingleOps, SingleRecordedOps,
        SingleRecordedState,
    },
    register_ops, TensorOpsAdd,
};
use num_traits::Float;

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorAddOps,
    forward |left, right| left * right,
    partial_left |state: &BinaryRecordedState<T, T, T>| state.right.clone(),
    partial_right |state: &BinaryRecordedState<T, T, T>| state.left.clone(),
);

register_ops!(
    ops SingleOps<T, T>,
    name ADTensorAddScalarOps state P,
    forward |state, input|  input * state,
    partial |state, state_recorded: &SingleRecordedState<T, T>|  state_recorded.input.ones() * state,
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
            tape self.tape.clone(),
            ops ADTensorAddOps::new(),
        );
        self.from_existing(node)
    }

    fn add_scalar(&self, other: &P) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsAdd::add_scalar(&self.tensor(), &other),
            tape self.tape.clone(),
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
