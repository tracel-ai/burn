use core::ops::Range;

use burn_common::stream::StreamId;

use crate::ops::IntTensorOps;
use crate::ops::{BoolTensor, FloatTensor, IntElem, IntTensor};
use crate::runner::{get_client, BackendRouter, RouterTensor, RunnerChannel, RunnerClient};
use crate::{Device, Distribution, Shape, TensorData};

impl<C: RunnerChannel> IntTensorOps<Self> for BackendRouter<C> {
    fn int_empty(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        todo!();
    }

    fn int_shape(tensor: &IntTensor<Self>) -> Shape {
        todo!();
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        tensor.into_data().await
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let client = get_client(&device);
        let id = StreamId::current();
        let desc = client.write_tensor(data, id);

        RouterTensor {
            desc,
            client,
            stream: id,
        }
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        todo!()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        todo!()
    }

    fn int_slice(tensor: IntTensor<Self>, ranges: &[Range<usize>]) -> IntTensor<Self> {
        todo!()
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        todo!()
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        todo!()
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        todo!()
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!()
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!()
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        todo!()
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        todo!()
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        todo!()
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        todo!()
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        todo!()
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        todo!()
    }
}
