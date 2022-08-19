use crate::tensor::api::Tensor;
use crate::{back::Backend, tensor::ops::*};
use crate::{
    define_ops, execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    Shape,
};
use num_traits::cast::FromPrimitive;
use rand::distributions::Standard;

define_ops! {
    name ADTensorOpsMean,
    state Shape<D>,
}

define_ops! {
    name ADTensorOpsSum,
    state Shape<D>,
}

impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<1>>
    for ADTensorOpsMean<B, D>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<1>>,
    ) -> B::TensorPrimitive<D> {
        let grad = state.output.grad();
        let ones = B::ones(self.state, grad.device());

        let grad: Tensor<B, 1> = Tensor::new(grad);
        let val = 1 as f64 / self.state.num_elements() as f64;
        let ones: Tensor<B, D> = Tensor::new(ones).mul_scalar(&B::Elem::from_f64(val).unwrap());

        ones.mul(&grad.unsqueeze()).value
    }
}

impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<1>>
    for ADTensorOpsSum<B, D>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<1>>,
    ) -> B::TensorPrimitive<D> {
        let grad = state.output.grad();
        let ones = B::ones(self.state, grad.device());

        let grad: Tensor<B, 1> = Tensor::new(grad);
        let ones: Tensor<B, D> = Tensor::new(ones);

        ones.mul(&grad.unsqueeze()).value
    }
}

macro_rules! define_impl {
    (
        $backend:ty,
        $backend_inner:ty,
        $element:ident
    ) => {
        impl<E: $element, const D: usize> TensorOpsAggregation<$backend, D>
            for <$backend as Backend>::TensorPrimitive<D>
        where
            Standard: rand::distributions::Distribution<E>,
        {
            fn mean(&self) -> <$backend as Backend>::TensorPrimitive<1> {
                execute_ops!(
                    input self.node.clone(),
                    out TensorOpsAggregation::mean(&self.tensor()),
                    ops ADTensorOpsMean::<$backend_inner, D>::new(self.shape.clone()),
                )
            }

            fn sum(&self) -> <$backend as Backend>::TensorPrimitive<1> {
                execute_ops!(
                    input self.node.clone(),
                    out TensorOpsAggregation::sum(&self.tensor()),
                    ops ADTensorOpsSum::<$backend_inner, D>::new(self.shape.clone()),
                )
            }

            fn mean_dim<const D2: usize>(
                &self,
                dim: usize,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                todo!()
            }

            fn sum_dim<const D2: usize>(
                &self,
                dim: usize,
            ) -> <$backend as Backend>::TensorPrimitive<D2> {
                todo!()
            }

            fn mean_dim_keepdim(&self, dim: usize) -> <$backend as Backend>::TensorPrimitive<D> {
                todo!()
            }

            fn sum_dim_keepdim(&self, dim: usize) -> <$backend as Backend>::TensorPrimitive<D> {
                todo!()
            }
        }
    };
}

crate::register_tch!();
crate::register_ndarray!();

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mean() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1.clone());
        let tensor_2 = TestADTensor::from_data(data_2.clone());

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_1.mul(&tensor_3.mean().unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[3.5, 9.5], [3.5, 9.5]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[-0.75, -0.75], [3.0, 3.0]]), 5);
    }
}
