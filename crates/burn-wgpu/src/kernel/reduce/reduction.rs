use crate::{
    codegen::{execute_dynamic, EagerHandle, WorkgroupLaunch},
    element::JitElement,
    kernel::reduce,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::Shape;

use super::{init_reduce_output, ArgMax, ArgMin, MeanDim, ReduceDim, ReduceDimEagerKernel, SumDim};

/// Sum all elements in the input buffer.
pub fn sum<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
) -> JitTensor<R, E, 1> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E, 1> = JitTensor::new(input.client, input.device, shape, input.handle);
    sum_dim(input, 0)
}

pub fn sum_dim<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    #[cfg(feature = "autotune")]
    {
        reduce::sum_dim_autotune(tensor, dim)
    }

    #[cfg(not(feature = "autotune"))]
    {
        let output = init_reduce_output(&tensor, dim);
        reduce::sum_dim_naive(tensor, output, dim)
    }
}

pub fn mean_dim<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    #[cfg(feature = "autotune")]
    {
        reduce::mean_dim_autotune(tensor, dim)
    }

    #[cfg(not(feature = "autotune"))]
    {
        let output = init_reduce_output(&tensor, dim);
        reduce::mean_dim_naive(tensor, output, dim)
    }
}

/// Execute the sum dim kernel.
pub fn sum_dim_naive<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_naive::<SumDim, R, E, E, D>(input, output, dim)
}

pub(crate) fn reduce_dim_naive<
    RD: ReduceDim,
    R: Runtime,
    EI: JitElement,
    EO: JitElement,
    const D: usize,
>(
    input: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    let kernel = ReduceDimEagerKernel::new(dim);

    execute_dynamic::<R, ReduceDimEagerKernel<RD, R, EI, EO>, EI>(
        &[EagerHandle::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        input.client,
    );

    output
}

/// Execute the int sum dim kernel.
pub fn int_sum_dim_naive<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_naive::<SumDim, R, E, E, D>(input, output, dim)
}

/// Execute the mean dim kernel.
pub fn mean_dim_naive<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_naive::<MeanDim, R, E, E, D>(input, output, dim)
}

/// Execute the int mean dim kernel.
pub fn int_mean_dim_naive<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    reduce_dim_naive::<MeanDim, R, E, E, D>(input, output, dim)
}

/// Execute the argmax kernel.
pub fn argmax<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, I, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let num_elems = shape_out.num_elements();
    let buffer = input.client.empty(num_elems * core::mem::size_of::<I>());
    let output = JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer,
    );
    reduce_dim_naive::<ArgMax, R, E, I, D>(input, output, dim)
}

/// Execute the argmin kernel.
pub fn argmin<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, I, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let num_elems = shape_out.num_elements();
    let buffer = input.client.empty(num_elems * core::mem::size_of::<I>());
    let output = JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer,
    );
    reduce_dim_naive::<ArgMin, R, E, I, D>(input, output, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kernel::reduce::init_reduce_output,
        tests::{ReferenceBackend, TestBackend, TestRuntime},
    };
    use burn_tensor::{ops::IntTensorOps, Data, Distribution, Int, Tensor};

    #[test]
    fn reduction_sum_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 1>::from_primitive(sum(tensor.into_primitive()));
        let val_ref = tensor_ref.sum();

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_sum_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let reduce_dim = 1;
        let output = init_reduce_output(&tensor.clone().into_primitive(), reduce_dim);

        let val =
            Tensor::<TestBackend, 2>::from_primitive(reduce_dim_naive::<
                SumDim,
                TestRuntime,
                f32,
                f32,
                2,
            >(
                tensor.into_primitive(), output, reduce_dim
            ));
        let val_ref = tensor_ref.sum_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_args_dim_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 2, Int>::from_primitive(argmax(tensor.into_primitive(), 1));
        let val_ref = tensor_ref.argmax(1);

        assert_eq!(val_ref.into_data().convert(), val.into_data());
    }

    #[test]
    fn sum_dim_should_work_with_int() {
        let summed_shape = Shape::new([1]);
        let data = Data::from([1, 2, 3, 4]);
        let tensor = TestBackend::int_from_data(data, &Default::default());

        let summed_tensor = TestBackend::int_empty(summed_shape, &Default::default());

        let val = Tensor::<TestBackend, 1, Int>::from_primitive(int_sum_dim_naive(
            tensor,
            summed_tensor,
            0,
        ));

        let sum_as_data = Data::from([10]);
        val.into_data().assert_approx_eq(&sum_as_data, 1);
    }

    #[test]
    fn mean_dim_should_work_with_int() {
        let mean_shape = Shape::new([1]);
        let data = Data::from([1, 2, 3, 4]);
        let tensor = TestBackend::int_from_data(data, &Default::default());

        let mean_tensor = TestBackend::int_empty(mean_shape, &Default::default());

        let val = Tensor::<TestBackend, 1, Int>::from_primitive(int_mean_dim_naive(
            tensor,
            mean_tensor,
            0,
        ));

        // Mean calculation truncates to an integer
        let mean_as_data = Data::from([2]);
        val.into_data().assert_approx_eq(&mean_as_data, 1);
    }
}
