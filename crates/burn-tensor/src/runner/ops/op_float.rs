use burn_common::stream::StreamId;

use crate::ops::{BoolTensor, FloatElem, FloatTensor, IntTensor};
use crate::repr::{FloatOperationDescription, OperationDescription, RandomOperationDescription};
use crate::runner::{Runner, RunnerClient, RunnerTensor};
use crate::{ops::FloatTensorOps, runner::RunnerBackend};
use crate::{Device, Distribution, Element, Shape, TensorData};
use std::ops::Range;

impl<B: RunnerBackend> FloatTensorOps<Self> for Runner<B> {
    fn float_from_data<const D: usize>(
        data: TensorData,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let client = B::client(&device);
        let stream = StreamId::current();
        let desc = client.write_tensor(data, stream);

        RunnerTensor {
            desc,
            client,
            stream,
        }
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let client = B::client(&device);
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

        RunnerTensor {
            desc,
            client,
            stream,
        }
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        todo!()
    }

    async fn float_into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> TensorData {
        tensor.into_data().await
    }

    fn float_device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        todo!()
    }

    fn float_to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        todo!()
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_remainder_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        todo!()
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        todo!()
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        todo!()
    }

    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        todo!()
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
    }

    fn float_sum_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
    }

    fn float_mean_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_exp<const D: usize>(lhs: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_powf_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: f32,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_cat<const D: usize>(
        tensors: Vec<FloatTensor<Self, D>>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_argmax<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn float_repeat_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_argmin<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        todo!()
    }

    fn float_max<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
    }

    fn float_max_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        todo!()
    }

    fn float_min<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
    }

    fn float_min_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        todo!()
    }

    fn float_powf<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_permute<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        todo!()
    }

    fn float_flip<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: &[usize],
    ) -> FloatTensor<Self, D> {
        todo!()
    }
}
