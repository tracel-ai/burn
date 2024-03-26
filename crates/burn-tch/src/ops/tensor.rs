use super::TchOps;
use crate::{element::TchElement, LibTorch, LibTorchDevice, TchShape, TchTensor};
use burn_tensor::{
    backend::Backend, ops::FloatTensorOps, Data, Distribution, ElementConversion, Reader, Shape,
};
use std::ops::Range;

impl<E: TchElement> FloatTensorOps<Self> for LibTorch<E> {
    fn float_from_data<const D: usize>(
        data: Data<E, D>,
        device: &LibTorchDevice,
    ) -> TchTensor<E, D> {
        TchTensor::from_data(data, (*device).into())
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &LibTorchDevice,
    ) -> TchTensor<E, D> {
        match distribution {
            Distribution::Default => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.rand_like_out(tensor))
                    .unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.uniform_(from, to)).unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn float_repeat<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        times: usize,
    ) -> TchTensor<E, D> {
        TchOps::repeat(tensor, dim, times)
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &LibTorchDevice) -> TchTensor<E, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(shape.dims, (E::KIND, device)))
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &LibTorchDevice) -> TchTensor<E, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(shape.dims, (E::KIND, device)))
    }

    fn float_shape<const D: usize>(
        tensor: &<LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
    ) -> Shape<D> {
        tensor.shape()
    }

    fn float_into_data<const D: usize>(
        tensor: <LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
    ) -> Reader<Data<<LibTorch<E> as Backend>::FloatElem, D>> {
        let shape = Self::float_shape(&tensor);
        let tensor = Self::float_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<E>, tch::TchError> = tensor.tensor.try_into();

        Reader::Concrete(Data::new(values.unwrap(), shape))
    }

    fn float_device<const D: usize>(tensor: &TchTensor<E, D>) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn float_to_device<const D: usize>(
        tensor: TchTensor<E, D>,
        device: &LibTorchDevice,
    ) -> TchTensor<E, D> {
        TchOps::to_device(tensor, device)
    }

    fn float_empty<const D: usize>(
        shape: Shape<D>,
        device: &<LibTorch<E> as Backend>::Device,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D> {
        let tensor = tch::Tensor::empty(shape.dims.map(|a| a as i64), (E::KIND, (*device).into()));

        TchTensor::new(tensor)
    }

    fn float_add<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::add(lhs, rhs)
    }

    fn float_add_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn float_sub<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::sub(lhs, rhs)
    }

    fn float_sub_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn float_mul<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::mul(lhs, rhs)
    }

    fn float_mul_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn float_div<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::div(lhs, rhs)
    }

    fn float_div_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        )
    }

    fn float_matmul<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = lhs.tensor.matmul(&rhs.tensor);
        TchTensor::new(tensor)
    }

    fn float_neg<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        TchTensor::new(tensor.tensor.reciprocal())
    }

    fn float_swap_dims<const D: usize>(
        tensor: TchTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> TchTensor<E, D> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<E, D2> {
        TchOps::reshape(tensor, shape)
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: TchTensor<E, D>,
        indices: TchTensor<i64, D>,
    ) -> TchTensor<E, D> {
        TchOps::gather(dim, tensor, indices)
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: TchTensor<E, D>,
        indices: TchTensor<i64, D>,
        value: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn float_select<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        indices: TchTensor<i64, 1>,
    ) -> TchTensor<E, D> {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn float_select_assign<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        indices: TchTensor<i64, 1>,
        value: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        ranges: [Range<usize>; D2],
    ) -> TchTensor<E, D1> {
        TchOps::slice(tensor, ranges)
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        ranges: [Range<usize>; D2],
        value: TchTensor<E, D1>,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D1> {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where<const D: usize>(
        tensor: TchTensor<E, D>,
        mask: TchTensor<bool, D>,
        value: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        let output = value.tensor.where_self(&mask.tensor, &tensor.tensor);

        TchTensor::new(output)
    }

    fn float_mask_fill<const D: usize>(
        tensor: TchTensor<E, D>,
        mask: TchTensor<bool, D>,
        value: E,
    ) -> TchTensor<E, D> {
        let value: f64 = value.elem();

        tensor.unary_ops(
            |mut tensor| tensor.f_masked_fill_(&mask.tensor, value).unwrap(),
            |tensor| tensor.f_masked_fill(&mask.tensor, value).unwrap(),
        )
    }

    fn float_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::equal(lhs, rhs)
    }

    fn float_equal_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_greater<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater(lhs, rhs)
    }

    fn float_greater_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::greater_elem(lhs, rhs.elem::<f64>())
    }

    fn float_greater_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: E,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_lower<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower(lhs, rhs)
    }

    fn float_lower_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::lower_elem(lhs, rhs.elem::<f64>())
    }

    fn float_lower_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::lower_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn float_mean<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        TchOps::mean(tensor)
    }

    fn float_sum<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        TchOps::sum(tensor)
    }

    fn float_sum_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::sum_dim(tensor, dim)
    }

    fn float_mean_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::mean_dim(tensor, dim)
    }

    fn float_prod<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        TchOps::prod(tensor)
    }

    fn float_prod_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::prod_dim(tensor, dim)
    }

    fn float_to_full_precision<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<f32, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(tch::Kind::Float);

        TchTensor::from_existing(tensor, storage)
    }

    fn float_from_full_precision<const D: usize>(tensor: TchTensor<f32, D>) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(E::KIND);

        TchTensor::from_existing(tensor, storage)
    }

    fn float_argmax<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::argmax(tensor, dim)
    }

    fn float_argmin<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::argmin(tensor, dim)
    }

    fn float_max_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::max_dim(tensor, dim)
    }

    fn float_max_dim_with_indices<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
    ) -> (TchTensor<E, D>, TchTensor<i64, D>) {
        TchOps::max_dim_with_indices(tensor, dim)
    }

    fn float_min_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::min_dim(tensor, dim)
    }

    fn float_min_dim_with_indices<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
    ) -> (TchTensor<E, D>, TchTensor<i64, D>) {
        TchOps::min_dim_with_indices(tensor, dim)
    }

    fn float_exp<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.exp_(), |tensor| tensor.exp())
    }

    fn float_log<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.log_(), |tensor| tensor.log())
    }

    fn float_log1p<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.log1p_(), |tensor| tensor.log1p())
    }

    fn float_powf_scalar<const D: usize>(tensor: TchTensor<E, D>, value: f32) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.f_pow_(value as f64).unwrap(),
            |tensor| tensor.pow_tensor_scalar(value as f64),
        )
    }

    fn float_sqrt<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.sqrt_(), |tensor| tensor.sqrt())
    }

    fn float_abs<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.abs_(), |tensor| tensor.abs())
    }

    fn float_cos<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.cos_(), |tensor| tensor.cos())
    }

    fn float_sin<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.sin_(), |tensor| tensor.sin())
    }

    fn float_tanh<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.tanh_(), |tensor| tensor.tanh())
    }

    fn float_erf<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.erf_(), |tensor| tensor.erf())
    }

    fn float_cat<const D: usize>(tensors: Vec<TchTensor<E, D>>, dim: usize) -> TchTensor<E, D> {
        TchOps::cat(tensors, dim)
    }

    fn float_clamp_min<const D: usize>(
        tensor: TchTensor<E, D>,
        min: E,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D> {
        TchOps::clamp_min(tensor, min.elem::<f64>())
    }

    fn float_clamp_max<const D: usize>(
        tensor: <LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
        max: <LibTorch<E> as Backend>::FloatElem,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D> {
        TchOps::clamp_max(tensor, max.elem::<f64>())
    }

    fn float_clamp<const D: usize>(
        tensor: <LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
        min: <LibTorch<E> as Backend>::FloatElem,
        max: <LibTorch<E> as Backend>::FloatElem,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D> {
        TchOps::clamp(tensor, min.elem::<f64>(), max.elem::<f64>())
    }

    fn float_into_int<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<i64, D> {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn float_narrow<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> TchTensor<E, D> {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn float_chunk<const D: usize>(
        tensor: TchTensor<E, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<TchTensor<E, D>> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn float_powf<const D: usize>(
        lhs: burn_tensor::ops::FloatTensor<Self, D>,
        rhs: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        TchOps::powf(lhs, rhs)
    }

    fn float_permute<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        TchOps::permute(tensor, axes)
    }

    fn float_flip<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
        axes: &[usize],
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        TchOps::flip(tensor, axes)
    }

    fn float_sign<const D: usize>(
        tensor: <LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D> {
        TchOps::sign(tensor)
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> burn_tensor::ops::FloatTensor<Self, D2> {
        TchOps::expand(tensor, shape)
    }

    fn float_sort<const D: usize>(
        tensor: <LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
        dim: usize,
        descending: bool,
    ) -> <LibTorch<E> as Backend>::FloatTensorPrimitive<D> {
        TchOps::sort(tensor, dim, descending)
    }

    fn float_argsort<const D: usize>(
        tensor: <LibTorch<E> as Backend>::FloatTensorPrimitive<D>,
        dim: usize,
        descending: bool,
    ) -> <LibTorch<E> as Backend>::IntTensorPrimitive<D> {
        TchOps::argsort(tensor, dim, descending)
    }
}
