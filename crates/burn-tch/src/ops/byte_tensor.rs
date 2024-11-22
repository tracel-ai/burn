use std::ops::Range;

use burn_tensor::{
    backend::Backend,
    ops::{ByteTensor, ByteTensorOps, IntTensorOps},
    Distribution, Shape, TensorData, TensorMetadata,
};

use crate::{element::TchElement, LibTorch, LibTorchDevice, QuantElement, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement, Q: QuantElement> ByteTensorOps<Self> for LibTorch<E, Q> {
    fn byte_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor {
        TchTensor::from_data::<i64>(data, (*device).into())
    }

    fn byte_repeat_dim(tensor: TchTensor, dim: usize, times: usize) -> TchTensor {
        TchOps::repeat_dim(tensor, dim, times)
    }

    async fn byte_into_data(tensor: TchTensor) -> TensorData {
        let shape = tensor.shape();
        let tensor = Self::int_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<i64>, tch::TchError> = tensor.tensor.shallow_clone().try_into();
        TensorData::new(values.unwrap(), shape)
    }

    fn byte_to_device(tensor: TchTensor, device: &LibTorchDevice) -> TchTensor {
        TchOps::to_device(tensor, device)
    }

    fn byte_reshape(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::reshape(tensor, shape)
    }

    fn byte_device(tensor: &TchTensor) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn byte_empty(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let tensor = tch::Tensor::empty(
            TchShape::from(shape).dims,
            (tch::Kind::Int64, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn byte_slice(tensor: TchTensor, ranges: &[Range<usize>]) -> TchTensor {
        TchOps::slice(tensor, ranges)
    }

    fn byte_slice_assign(
        tensor: TchTensor,
        ranges: &[Range<usize>],
        value: TchTensor,
    ) -> TchTensor {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn byte_cat(tensors: Vec<TchTensor>, dim: usize) -> TchTensor {
        TchOps::cat(tensors, dim)
    }

    fn byte_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::equal(lhs, rhs)
    }

    fn byte_equal_elem(lhs: TchTensor, rhs: u8) -> TchTensor {
        TchOps::equal_elem(lhs, rhs as i64)
    }

    fn byte_greater(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::greater(lhs, rhs)
    }

    fn byte_greater_elem(lhs: TchTensor, rhs: u8) -> TchTensor {
        TchOps::greater_elem(lhs, rhs as i64)
    }

    fn byte_greater_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::greater_equal(lhs, rhs)
    }

    fn byte_greater_equal_elem(lhs: TchTensor, rhs: u8) -> TchTensor {
        TchOps::greater_equal_elem(lhs, rhs as i64)
    }

    fn byte_lower(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::lower(lhs, rhs)
    }

    fn byte_lower_elem(lhs: TchTensor, rhs: u8) -> TchTensor {
        TchOps::lower_elem(lhs, rhs as i64)
    }

    fn byte_lower_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::lower_equal(lhs, rhs)
    }

    fn byte_lower_equal_elem(lhs: TchTensor, rhs: u8) -> TchTensor {
        TchOps::lower_equal_elem(lhs, rhs as i64)
    }

    fn byte_add(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::add(lhs, rhs)
    }

    fn byte_add_scalar(lhs: TchTensor, rhs: u8) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs as i64).unwrap(),
            |tensor| tensor.f_add_scalar(rhs as i64).unwrap(),
        )
    }

    fn byte_sub(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::sub(lhs, rhs)
    }

    fn byte_sub_scalar(lhs: TchTensor, rhs: u8) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs as i64).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs as i64).unwrap(),
        )
    }

    fn byte_mul(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::mul(lhs, rhs)
    }

    fn byte_mul_scalar(lhs: TchTensor, rhs: u8) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs as i64).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs as i64).unwrap(),
        )
    }

    fn byte_div(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));
        let rhs: TchTensor =
            TchTensor::new(rhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out = TchOps::div(lhs, rhs);

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn byte_div_scalar(lhs: TchTensor, rhs: u8) -> TchTensor {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out: TchTensor = lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs as i64).unwrap(),
            |tensor| tensor.f_div_scalar(rhs as i64).unwrap(),
        );

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn byte_remainder(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        let copy = false;
        let non_blocking = true;
        let lhs: TchTensor =
            TchTensor::new(lhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));
        let rhs: TchTensor =
            TchTensor::new(rhs.tensor.to_dtype(tch::Kind::Float, non_blocking, copy));

        let out = TchOps::remainder(lhs, rhs);

        TchTensor::new(out.tensor.to_dtype(tch::Kind::Int64, non_blocking, copy))
    }

    fn byte_remainder_scalar(lhs: TchTensor, rhs: u8) -> TchTensor {
        lhs.unary_ops(
            |tensor| tensor.f_remainder(rhs as i64).unwrap(),
            |tensor| tensor.f_remainder(rhs as i64).unwrap(),
        )
    }

    fn byte_neg(tensor: TchTensor) -> TchTensor {
        Self::int_mul_scalar(tensor, -1)
    }

    fn byte_zeros(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (tch::Kind::Int64, device)))
    }

    fn byte_ones(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (tch::Kind::Int64, device)))
    }

    fn byte_full(
        shape: Shape,
        fill_value: u8,
        device: &<LibTorch<E> as Backend>::Device,
    ) -> TchTensor {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::full(
            shape.dims,
            fill_value as i64,
            (tch::Kind::Int64, device),
        ))
    }

    fn byte_sum(tensor: TchTensor) -> TchTensor {
        TchOps::sum(tensor)
    }

    fn byte_sum_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::sum_dim(tensor, dim)
    }

    fn byte_prod(tensor: TchTensor) -> TchTensor {
        TchOps::prod(tensor)
    }

    fn byte_prod_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::prod_dim(tensor, dim)
    }

    fn byte_mean(tensor: TchTensor) -> TchTensor {
        let tensor: TchTensor =
            TchTensor::new(tensor.tensor.to_dtype(tch::Kind::Float, true, false));
        let output: TchTensor = TchTensor::new(TchOps::mean(tensor).tensor);

        TchTensor::new(output.tensor.to_dtype(tch::Kind::Int64, true, false))
    }

    fn byte_mean_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        let tensor: TchTensor =
            TchTensor::new(tensor.tensor.to_dtype(tch::Kind::Float, true, false));

        let output: TchTensor = TchTensor::new(TchOps::mean_dim(tensor, dim).tensor);

        TchTensor::new(output.tensor.to_dtype(tch::Kind::Int64, true, false))
    }

    fn byte_gather(dim: usize, tensor: TchTensor, indices: TchTensor) -> TchTensor {
        TchOps::gather(dim, tensor, indices)
    }

    fn byte_scatter(
        dim: usize,
        tensor: TchTensor,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn byte_select(tensor: TchTensor, dim: usize, indices: TchTensor) -> TchTensor {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn byte_select_assign(
        tensor: TchTensor,
        dim: usize,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn byte_mask_where(tensor: TchTensor, mask: TchTensor, source: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            tensor,
            source,
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
        )
    }

    fn byte_mask_fill(tensor: TchTensor, mask: TchTensor, value: u8) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value as i64).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value as i64).unwrap(),
        )
    }

    fn byte_argmax(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::argmax(tensor, dim)
    }

    fn byte_argmin(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::argmin(tensor, dim)
    }

    fn byte_max_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::max_dim(tensor, dim)
    }

    fn byte_max_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn byte_min_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchOps::min_dim(tensor, dim)
    }

    fn byte_min_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn byte_clamp_min(tensor: TchTensor, min: u8) -> TchTensor {
        TchOps::clamp_min(tensor, min as i64)
    }

    fn byte_clamp_max(tensor: TchTensor, max: u8) -> TchTensor {
        TchOps::clamp_max(tensor, max as i64)
    }

    fn byte_clamp(tensor: TchTensor, min: u8, max: u8) -> TchTensor {
        TchOps::clamp(tensor, min as i64, max as i64)
    }

    fn byte_abs(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn byte_into_float(tensor: TchTensor) -> TchTensor {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }

    fn byte_into_int(tensor: TchTensor) -> TchTensor {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn byte_swap_dims(tensor: ByteTensor<Self>, dim1: usize, dim2: usize) -> ByteTensor<Self> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn byte_narrow(tensor: TchTensor, dim: usize, start: usize, length: usize) -> TchTensor {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn byte_chunk(tensor: TchTensor, chunks: usize, dim: usize) -> Vec<TchTensor> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn byte_split(tensor: TchTensor, split_size: usize, dim: usize) -> Vec<TchTensor> {
        TchOps::split(tensor, split_size, dim)
    }

    fn byte_split_with_sizes(
        tensor: TchTensor,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<TchTensor> {
        TchOps::split_with_sizes(tensor, split_sizes, dim)
    }

    fn byte_random(shape: Shape, distribution: Distribution, device: &LibTorchDevice) -> TchTensor {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.uniform_(0.0, 255.0))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::empty::<i64>(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn byte_arange(range: Range<i64>, device: &LibTorchDevice) -> TchTensor {
        let device: tch::Device = (*device).into();
        let mut tensor = tch::Tensor::arange(range.end - range.start, (tch::Kind::Int64, device));

        if range.start != 0 {
            tensor = tensor.f_add_scalar_(range.start).unwrap();
        }

        TchTensor::new(tensor)
    }

    fn byte_permute(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        TchOps::permute(tensor, axes)
    }

    fn byte_flip(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        TchOps::flip(tensor, axes)
    }

    fn byte_sign(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        TchOps::sign(tensor)
    }

    fn byte_expand(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        TchOps::expand(tensor, shape)
    }

    fn byte_sort(tensor: ByteTensor<Self>, dim: usize, descending: bool) -> ByteTensor<Self> {
        TchOps::sort(tensor, dim, descending)
    }

    fn byte_argsort(tensor: ByteTensor<Self>, dim: usize, descending: bool) -> ByteTensor<Self> {
        TchOps::argsort(tensor, dim, descending)
    }
}
