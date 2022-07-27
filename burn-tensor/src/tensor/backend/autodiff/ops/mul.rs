use crate::{
    execute_ops,
    graph::ops::{BinaryOps, BinaryOpsNodeState, UnaryOps, UnaryOpsNodeState},
    register_ops,
    tensor::{
        backend::autodiff::{ADCompatibleTensor, ADElement, ADTensor},
        ops::*,
    },
};

register_ops!(
    ops BinaryOps<T, T, T>,
    name ADTensorMulOps,
    partial_left |state: &BinaryOpsNodeState<T, T, T>| {
        state.output.grad() * state.right.value().clone()
    },
    partial_right |state: &BinaryOpsNodeState<T, T, T>| {
        state.output.grad() * state.left.value().clone()
    },
);

register_ops!(
    ops UnaryOps<T, T>,
    name ADTensorMulScalarOps state P,
    partial |state, state_recorded: &UnaryOpsNodeState<T, T>| {
        state_recorded.output.grad() * state
    },
);

impl<T, P, const D: usize> TensorOpsMul<P, D> for ADTensor<P, D, T>
where
    T: ADCompatibleTensor<P, D>,
    P: ADElement,
{
    fn mul(&self, other: &Self) -> Self {
        let node = execute_ops!(
            lhs self.node.clone(),
            rhs other.node.clone(),
            out TensorOpsMul::mul(&self.tensor(), &other.tensor()),
            ops ADTensorMulOps::new(),
        );
        self.from_existing(node)
    }

    fn mul_scalar(&self, other: &P) -> Self {
        let node = execute_ops!(
            input self.node.clone(),
            out TensorOpsMul::mul_scalar(&self.tensor(), &other),
            ops ADTensorMulScalarOps::new(other.clone()),
        );
        self.from_existing(node)
    }
}

impl<T, P, const D: usize> std::ops::Mul<P> for ADTensor<P, D, T>
where
    T: ADCompatibleTensor<P, D> + 'static,
    P: ADElement + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn mul(self, rhs: P) -> Self::Output {
        TensorOpsMul::mul_scalar(&self, &rhs)
    }
}

impl<T, P, const D: usize> std::ops::Mul<ADTensor<P, D, T>> for ADTensor<P, D, T>
where
    T: ADCompatibleTensor<P, D> + 'static,
    P: ADElement + 'static,
{
    type Output = ADTensor<P, D, T>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOpsMul::mul(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{backend::autodiff::helper::ADTchTensor, Data};

    #[test]
    fn should_diff_mul() {
        let data_1 = Data::from([1.0, 7.0]);
        let data_2 = Data::from([4.0, 7.0]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.clone() * tensor_2.clone();
        let grads = tensor_3.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(grad_1.to_data(), data_2);
        assert_eq!(grad_2.to_data(), data_1);
        assert_eq!(tensor_3.into_data(), Data::from([4.0, 49.0]));
    }

    #[test]
    fn should_diff_mul_scalar() {
        let data = Data::from([2.0, 5.0]);

        let tensor = ADTchTensor::from_data(data.clone());
        let tensor_out = tensor.clone() * 4.0;

        let grads = tensor_out.backward();
        let grad = grads.wrt(&tensor).unwrap();

        assert_eq!(tensor_out.into_data(), Data::from([8.0, 20.0]));
        assert_eq!(grad.to_data(), Data::from([4.0, 4.0]));
    }

    #[test]
    fn test_mul_complex_1() {
        let data_1: Data<f64, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f64, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f64, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = ADTchTensor::from_data(data_1.clone());
        let tensor_2 = ADTchTensor::from_data(data_2.clone());
        let tensor_3 = ADTchTensor::from_data(data_3.clone());

        let tensor_4 = tensor_1.mul(&tensor_2);
        let tensor_5 = tensor_4.mul(&tensor_3);
        let tensor_6 = tensor_1.mul(&tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = grads.wrt(&tensor_1).unwrap();
        let grad_2 = grads.wrt(&tensor_2).unwrap();

        assert_eq!(
            grad_1.to_data(),
            Data::from([[16.0, 196.0], [104.0, -36.0]])
        );
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 98.0], [338.0, 18.0]]));
    }
}
