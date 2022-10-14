use crate::backend::autodiff::ADBackendDecorator;
use crate::tensor::ElementConversion;
use crate::Tensor;
use crate::{backend::Backend, tensor::ops::*};
use crate::{
    define_ops, execute_ops,
    graph::ops::{UnaryOps, UnaryOpsNodeState},
    Shape,
};

define_ops! {
    name ADTensorOpsMean,
    state Shape<D>,
}

define_ops! {
    name ADTensorOpsSum,
    state Shape<D>,
}

define_ops! {
    name ADTensorOpsMeanDim,
    state (Shape<D>, usize),
}

define_ops! {
    name ADTensorOpsSumDim,
    state (Shape<D>, usize),
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
        let val = 1_f64 / self.state.num_elements() as f64;
        let ones: Tensor<B, D> = Tensor::new(ones).mul_scalar(val);

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

impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
    for ADTensorOpsMeanDim<B, D>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
    ) -> B::TensorPrimitive<D> {
        let (shape, dim) = self.state;

        let grad = state.output.grad().sum_dim(dim);
        let ones = B::ones(shape, grad.device());

        let val = 1_f64 / shape.dims[dim] as f64;
        let ones = ones.mul_scalar(&B::Elem::from_elem(val));

        ones.mul(&grad)
    }
}

impl<B: Backend, const D: usize> UnaryOps<B::TensorPrimitive<D>, B::TensorPrimitive<D>>
    for ADTensorOpsSumDim<B, D>
{
    fn partial(
        &self,
        state: &UnaryOpsNodeState<B::TensorPrimitive<D>, B::TensorPrimitive<D>>,
    ) -> B::TensorPrimitive<D> {
        let (shape, dim) = self.state;

        let grad = state.output.grad().sum_dim(dim);
        let ones = B::ones(shape, grad.device());

        ones.mul(&grad)
    }
}

impl<B: Backend, const D: usize> TensorOpsAggregation<ADBackendDecorator<B>, D>
    for <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>
{
    fn mean(&self) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<1> {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsAggregation::mean(&self.tensor()),
            ops ADTensorOpsMean::<B, D>::new(self.shape),
        )
    }

    fn sum(&self) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<1> {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsAggregation::sum(&self.tensor()),
            ops ADTensorOpsSum::<B, D>::new(self.shape),
        )
    }

    fn mean_dim(&self, dim: usize) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsAggregation::mean_dim(&self.tensor(), dim),
            ops ADTensorOpsMeanDim::<B, D>::new((self.shape, dim)),
        )
    }

    fn sum_dim(&self, dim: usize) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        execute_ops!(
            input self.node.clone(),
            out TensorOpsAggregation::sum_dim(&self.tensor(), dim),
            ops ADTensorOpsSumDim::<B, D>::new((self.shape, dim)),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{backend::autodiff::helper::TestADTensor, Data};

    #[test]
    fn should_diff_mean() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

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

    #[test]
    fn should_diff_sum() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_1.mul(&tensor_3.sum().unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[14.0, 38.0], [14.0, 38.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[-3.0, -3.0], [12.0, 12.0]]), 5);
    }

    #[test]
    fn should_diff_mean_dim() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_1.mul(&tensor_3.mean_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[4.0, 36.0], [3.0, -17.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[9.0, 9.0], [35.5, 35.5]]), 5);
    }

    #[test]
    fn should_diff_sum_dim() {
        let data_1 = Data::<f64, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f64, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);

        let tensor_3 = tensor_1.matmul(&tensor_2);
        let tensor_4 = tensor_1.mul(&tensor_3.sum_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[8.0, 72.0], [6.0, -34.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[18.0, 18.0], [71.0, 71.0]]), 5);
    }
}
