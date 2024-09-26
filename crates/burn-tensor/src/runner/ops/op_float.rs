use burn_common::stream::StreamId;

use crate::ops::FloatTensorOps;
use crate::ops::{BoolTensor, FloatElem, FloatTensor, IntTensor};
use crate::repr::{FloatOperationDescription, OperationDescription, RandomOperationDescription};
use crate::runner::{BackendRouter, MultiBackendRuntime, RouterTensor, RunnerClient};
use crate::{Device, Distribution, Element, Shape, TensorData};
use std::ops::Range;

impl<R: MultiBackendRuntime> FloatTensorOps<Self> for BackendRouter<R> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let client = R::client(&device);
        let stream = StreamId::current();
        let desc = client.write_tensor(data, stream);

        RouterTensor {
            desc,
            client,
            stream,
            runner_id,
        }
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = R::client(&device);
        let stream = StreamId::current();
        let dtype = FloatElem::<Self>::dtype();
        let desc = client.empty_tensor(shape.dims.to_vec(), dtype, stream);

        client.register(
            OperationDescription::Float(
                dtype,
                FloatOperationDescription::Random(RandomOperationDescription {
                    out: desc.clone(),
                    distribution,
                }),
            ),
            stream,
        );

        RouterTensor {
            desc,
            client,
            stream,
        }
    }

    fn float_zeros(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_ones(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_shape(tensor: &FloatTensor<Self>) -> Shape {
        todo!()
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        tensor.into_data().await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        todo!()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        todo!()
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_slice(tensor: FloatTensor<Self>, ranges: &[Range<usize>]) -> FloatTensor<Self> {
        todo!()
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_exp(lhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_powf_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        todo!()
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        todo!()
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        todo!()
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        todo!()
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        todo!()
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        todo!()
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        todo!()
    }
}
