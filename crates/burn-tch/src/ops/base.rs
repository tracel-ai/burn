use burn_tensor::{Shape, TensorMetadata};
use tch::Scalar;

use crate::{LibTorchDevice, TchShape, TchTensor};
use std::ops::Range;

pub struct TchOps {
    // e: PhantomData<E>,
}

impl TchOps {
    pub fn to_device(tensor: TchTensor, device: &LibTorchDevice) -> TchTensor {
        let device = (*device).into();

        // We have to manually check if the device is the same, since when it's the case, we need to keep
        // the same storage reference and not create a new one.
        if tensor.tensor.device() == device {
            return tensor;
        }

        TchTensor::new(tensor.tensor.to(device))
    }

    pub fn reshape(tensor: TchTensor, shape: Shape) -> TchTensor {
        let shape_tch: TchShape = shape.into();

        TchTensor::from_existing(tensor.tensor.reshape(shape_tch.dims), tensor.storage)
    }

    pub fn repeat_dim(tensor: TchTensor, dim: usize, times: usize) -> TchTensor {
        let mut dims = vec![1; tensor.shape().num_dims()];
        dims[dim] = times as i64;
        let tensor = tch::Tensor::repeat(&tensor.tensor, dims);
        TchTensor::new(tensor)
    }

    pub fn slice(tensor: TchTensor, ranges: &[Range<usize>]) -> TchTensor {
        let storage = tensor.storage.clone();
        let mut tensor = tensor.tensor.shallow_clone();

        for (i, index) in ranges.iter().enumerate().take(ranges.len()) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;
            tensor = tensor.narrow(i as i64, start, length);
        }

        TchTensor::partial(tensor, storage)
    }

    pub fn slice_assign(tensor: TchTensor, ranges: &[Range<usize>], value: TchTensor) -> TchTensor {
        let tch_shape = TchShape::from(tensor.shape());

        // Copy the input tensor if we can't mutate it.
        let tensor_original: TchTensor = tensor.unary_ops(|tensor| tensor, |tensor| tensor.copy());
        let tensor_original = tensor_original.tensor;

        let mut tensor = tensor_original.view_(tch_shape.dims);

        for (i, index) in ranges.iter().enumerate().take(ranges.len()) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;

            tensor = tensor.narrow(i as i64, start, length);
        }

        tensor.copy_(&value.tensor);

        TchTensor::new(tensor_original)
    }

    pub fn gather(dim: usize, tensor: TchTensor, indices: TchTensor) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.gather(dim as i64, &indices.tensor, false);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn scatter(
        dim: usize,
        tensor: TchTensor,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor
            .tensor
            .scatter_add(dim as i64, &indices.tensor, &value.tensor);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn index_select_dim(tensor: TchTensor, dim: usize, indices: TchTensor) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.index_select(dim as i64, &indices.tensor);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn select_assign(
        tensor: TchTensor,
        dim: usize,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        tensor.clone().unary_ops(
            |mut tensor| tensor.index_add_(dim as i64, &indices.tensor, &value.tensor),
            |tensor| tensor.index_add(dim as i64, &indices.tensor, &value.tensor),
        )
    }

    pub fn cat(tensors: Vec<TchTensor>, dim: usize) -> TchTensor {
        let tensors: Vec<tch::Tensor> = tensors
            .into_iter()
            .map(|t| t.tensor.shallow_clone())
            .collect();
        let tensor = tch::Tensor::cat(&tensors, dim as i64);

        TchTensor::new(tensor)
    }

    pub fn equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.eq_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.eq_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.eq_tensor(rhs),
        )
    }

    pub fn equal_elem<S: Into<tch::Scalar> + Clone>(lhs: TchTensor, rhs: S) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.eq_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(rhs.clone().into()),
        )
    }

    pub fn greater(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.greater_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.less_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.greater_tensor(rhs),
        )
    }

    pub fn greater_elem<S: Into<tch::Scalar> + Clone>(lhs: TchTensor, rhs: S) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.greater_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.greater(rhs.clone().into()),
        )
    }

    pub fn greater_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.greater_equal_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.less_equal_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.greater_equal_tensor(rhs),
        )
    }

    pub fn greater_equal_elem<S: Into<Scalar> + Clone>(lhs: TchTensor, rhs: S) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| {
                tensor
                    .greater_equal_(rhs.clone().into())
                    .to_kind(tch::Kind::Bool)
            },
            |tensor| tensor.greater_equal(rhs.clone().into()),
        )
    }

    pub fn lower(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.less_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.greater_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.less_tensor(rhs),
        )
    }

    pub fn lower_elem<S: Into<Scalar> + Clone>(lhs: TchTensor, rhs: S) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| tensor.less_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.less(rhs.clone().into()),
        )
    }

    pub fn lower_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.less_equal_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.greater_equal_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.less_equal_tensor(rhs),
        )
    }

    pub fn lower_equal_elem<S: Into<Scalar> + Clone>(lhs: TchTensor, rhs: S) -> TchTensor {
        lhs.unary_ops(
            |mut tensor| {
                tensor
                    .less_equal_(rhs.clone().into())
                    .to_kind(tch::Kind::Bool)
            },
            |tensor| tensor.less_equal(rhs.clone().into()),
        )
    }

    pub fn add(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_add_(rhs).unwrap(),
            |lhs, rhs| rhs.f_add_(lhs).unwrap(),
            |lhs, rhs| lhs.f_add(rhs).unwrap(),
        )
    }

    pub fn sub(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_sub_(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
        )
    }

    pub fn mul(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_mul_(rhs).unwrap(),
            |lhs, rhs| rhs.f_mul_(lhs).unwrap(),
            |lhs, rhs| lhs.f_mul(rhs).unwrap(),
        )
    }

    pub fn div(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_div_(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
        )
    }

    pub fn remainder(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_remainder_tensor_(rhs).unwrap(),
            |lhs, rhs| lhs.f_remainder_tensor(rhs).unwrap(),
            |lhs, rhs| lhs.f_remainder_tensor(rhs).unwrap(),
        )
    }

    pub fn mean(tensor: TchTensor) -> TchTensor {
        // view as 1d tensor
        let tensor = tensor.tensor.mean(tensor.tensor.kind()).view(1);
        TchTensor::new(tensor)
    }

    pub fn mean_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchTensor::from_existing(
            tensor
                .tensor
                .mean_dim(Some([dim as i64].as_slice()), true, tensor.tensor.kind()),
            tensor.storage,
        )
    }

    pub fn sum(tensor: TchTensor) -> TchTensor {
        // view as 1d tensor
        let tensor = tensor.tensor.sum(tensor.tensor.kind()).view(1);
        TchTensor::new(tensor)
    }

    pub fn sum_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchTensor::from_existing(
            tensor.tensor.sum_dim_intlist(
                Some([dim as i64].as_slice()),
                true,
                tensor.tensor.kind(),
            ),
            tensor.storage,
        )
    }

    pub fn prod(tensor: TchTensor) -> TchTensor {
        // view as 1d tensor
        let tensor = tensor.tensor.prod(tensor.tensor.kind()).view(1);
        TchTensor::new(tensor)
    }

    pub fn prod_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        TchTensor::from_existing(
            tensor
                .tensor
                .prod_dim_int(dim as i64, true, tensor.tensor.kind()),
            tensor.storage,
        )
    }

    pub fn argmax(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmax(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn argmin(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmin(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn max_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let (tensor, _indices) = tensor.tensor.max_dim(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn max_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        let storage = tensor.storage.clone();
        let (tensor, indices) = tensor.tensor.max_dim(dim as i64, true);

        let tensor = TchTensor::from_existing(tensor, storage);
        let indices = TchTensor::new(indices);

        (tensor, indices)
    }

    pub fn min_dim(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let (tensor, _indices) = tensor.tensor.min_dim(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn min_dim_with_indices(tensor: TchTensor, dim: usize) -> (TchTensor, TchTensor) {
        let storage = tensor.storage.clone();
        let (tensor, indices) = tensor.tensor.min_dim(dim as i64, true);

        let tensor = TchTensor::from_existing(tensor, storage);
        let indices = TchTensor::new(indices);

        (tensor, indices)
    }

    pub fn clamp_min<S: Into<tch::Scalar> + Clone + Copy>(tensor: TchTensor, min: S) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_min_(min),
            |tensor| tensor.clamp_min(min),
        )
    }

    pub fn clamp_max<S: Into<tch::Scalar> + Clone + Copy>(tensor: TchTensor, max: S) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_max_(max),
            |tensor| tensor.clamp_max(max),
        )
    }

    pub fn clamp<S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor,
        min: S,
        max: S,
    ) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_(min, max),
            |tensor| tensor.clamp(min, max),
        )
    }

    pub fn swap_dims(tensor: TchTensor, dim1: usize, dim2: usize) -> TchTensor {
        let tensor = tensor.tensor.transpose(dim1 as i64, dim2 as i64);
        TchTensor::new(tensor)
    }

    pub fn permute(tensor: TchTensor, axes: &[usize]) -> TchTensor {
        let tensor = tensor
            .tensor
            .permute(axes.iter().map(|x| *x as i64).collect::<Vec<_>>());
        TchTensor::new(tensor)
    }

    pub fn flip(tensor: TchTensor, axes: &[usize]) -> TchTensor {
        let dims = axes.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let tensor = tensor.tensor.flip(dims);
        TchTensor::new(tensor)
    }

    pub fn narrow(tensor: TchTensor, dim: usize, start: usize, length: usize) -> TchTensor {
        TchTensor::new(
            tensor
                .tensor
                .narrow(dim as i64, start as i64, length as i64),
        )
    }

    pub fn chunk(tensor: TchTensor, chunks: usize, dim: usize) -> Vec<TchTensor> {
        tensor
            .tensor
            .chunk(chunks as i64, dim as i64)
            .into_iter()
            .map(TchTensor::new)
            .collect()
    }

    pub fn split(tensor: TchTensor, split_size: usize, dim: usize) -> Vec<TchTensor> {
        tensor
            .tensor
            .split(split_size as i64, dim as i64)
            .into_iter()
            .filter(|x| x.numel() > 0)
            .map(TchTensor::new)
            .collect()
    }

    pub fn split_with_sizes(
        tensor: TchTensor,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<TchTensor> {
        let split_sizes_i64: Vec<i64> = split_sizes.iter().map(|&s| s as i64).collect();
        tensor
            .tensor
            .split_with_sizes(split_sizes_i64, dim as i64)
            .into_iter()
            .filter(|x| x.numel() > 0)
            .map(TchTensor::new)
            .collect()
    }

    pub fn powf(tensor: TchTensor, exponent: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            tensor,
            exponent,
            |lhs, rhs| lhs.f_pow_tensor_(rhs).unwrap(),
            |lhs, rhs| lhs.f_pow(rhs).unwrap(),
            |lhs, rhs| lhs.f_pow(rhs).unwrap(),
        )
    }

    pub fn sign(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.sign_(), |tensor| tensor.sign())
    }

    pub fn expand(tensor: TchTensor, shape: Shape) -> TchTensor {
        let storage = tensor.storage.clone();
        let broadcasted_tensor = tensor.tensor.broadcast_to(TchShape::from(shape).dims);
        TchTensor::from_existing(broadcasted_tensor, storage)
    }

    pub fn sort(tensor: TchTensor, dim: usize, descending: bool) -> TchTensor {
        TchTensor::new(tensor.tensor.sort(dim as i64, descending).0)
    }

    pub fn sort_with_indices(
        tensor: TchTensor,
        dim: usize,
        descending: bool,
    ) -> (TchTensor, TchTensor) {
        let sorted = tensor.tensor.sort(dim as i64, descending);
        (TchTensor::new(sorted.0), TchTensor::new(sorted.1))
    }

    pub fn argsort(tensor: TchTensor, dim: usize, descending: bool) -> TchTensor {
        TchTensor::new(tensor.tensor.argsort(dim as i64, descending))
    }

    pub fn bitwise_and(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_bitwise_and_tensor_(rhs).unwrap(),
            |lhs, rhs| rhs.f_bitwise_and_tensor_(lhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_and_tensor(rhs).unwrap(),
        )
    }

    pub fn bitwise_and_scalar<S: Into<Scalar> + Clone>(tensor: TchTensor, scalar: S) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_bitwise_and_(scalar.clone().into()).unwrap(),
            |tensor| tensor.f_bitwise_and(scalar.clone().into()).unwrap(),
        )
    }

    pub fn bitwise_or(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_bitwise_or_tensor_(rhs).unwrap(),
            |lhs, rhs| rhs.f_bitwise_or_tensor_(lhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_or_tensor(rhs).unwrap(),
        )
    }

    pub fn bitwise_or_scalar<S: Into<Scalar> + Clone>(tensor: TchTensor, scalar: S) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_bitwise_or_(scalar.clone().into()).unwrap(),
            |tensor| tensor.f_bitwise_or(scalar.clone().into()).unwrap(),
        )
    }

    pub fn bitwise_xor(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_bitwise_xor_tensor_(rhs).unwrap(),
            |lhs, rhs| rhs.f_bitwise_xor_tensor_(lhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_xor_tensor(rhs).unwrap(),
        )
    }

    pub fn bitwise_xor_scalar<S: Into<Scalar> + Clone>(tensor: TchTensor, scalar: S) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_bitwise_xor_(scalar.clone().into()).unwrap(),
            |tensor| tensor.f_bitwise_xor(scalar.clone().into()).unwrap(),
        )
    }

    pub fn bitwise_not(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.f_bitwise_not_().unwrap(),
            |tensor| tensor.f_bitwise_not().unwrap(),
        )
    }

    pub fn bitwise_left_shift(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_bitwise_left_shift_(rhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_left_shift(rhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_left_shift(rhs).unwrap(),
        )
    }

    pub fn bitwise_left_shift_scalar<S: Into<Scalar> + Clone>(
        tensor: TchTensor,
        scalar: S,
    ) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| {
                tensor
                    .f_bitwise_left_shift_tensor_scalar_(scalar.clone().into())
                    .unwrap()
            },
            |tensor| {
                tensor
                    .f_bitwise_left_shift_tensor_scalar(scalar.clone().into())
                    .unwrap()
            },
        )
    }

    pub fn bitwise_right_shift(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_bitwise_right_shift_(rhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_right_shift(rhs).unwrap(),
            |lhs, rhs| lhs.f_bitwise_right_shift(rhs).unwrap(),
        )
    }

    pub fn bitwise_right_shift_scalar<S: Into<Scalar> + Clone>(
        tensor: TchTensor,
        scalar: S,
    ) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| {
                tensor
                    .f_bitwise_right_shift_tensor_scalar_(scalar.clone().into())
                    .unwrap()
            },
            |tensor| {
                tensor
                    .f_bitwise_right_shift_tensor_scalar(scalar.clone().into())
                    .unwrap()
            },
        )
    }
}
