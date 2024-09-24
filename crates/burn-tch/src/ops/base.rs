use burn_tensor::{quantization::QuantizationStrategy, Shape};
use tch::Scalar;

use crate::{LibTorchDevice, TchShape, TchTensor};
use std::{marker::PhantomData, ops::Range};

pub struct TchOps<E: tch::kind::Element + Copy + Default> {
    e: PhantomData<E>,
}

impl<E: tch::kind::Element + Copy + Default> TchOps<E> {
    pub fn to_device(tensor: TchTensor<E>, device: &LibTorchDevice) -> TchTensor<E> {
        let device = (*device).into();

        // We have to manually check if the device is the same, since when it's the case, we need to keep
        // the same storage reference and not create a new one.
        if tensor.tensor.device() == device {
            return tensor;
        }

        TchTensor::new(tensor.tensor.to(device))
    }

    pub fn reshape(tensor: TchTensor<E>, shape: Shape) -> TchTensor<E> {
        let shape_tch: TchShape = shape.into();

        TchTensor::from_existing(tensor.tensor.reshape(shape_tch.dims), tensor.storage)
    }

    pub fn repeat_dim(tensor: TchTensor<E>, dim: usize, times: usize) -> TchTensor<E> {
        let mut dims = vec![1; tensor.shape().num_dims()];
        dims[dim] = times as i64;
        let tensor = tch::Tensor::repeat(&tensor.tensor, dims);
        TchTensor::new(tensor)
    }

    pub fn slice(tensor: TchTensor<E>, ranges: &[Range<usize>]) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let mut tensor = tensor.tensor.shallow_clone();

        for (i, index) in ranges.iter().enumerate().take(ranges.len()) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;
            tensor = tensor.narrow(i as i64, start, length);
        }

        TchTensor::partial(tensor, storage)
    }

    pub fn slice_assign(
        tensor: TchTensor<E>,
        ranges: &[Range<usize>],
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        let tch_shape = TchShape::from(tensor.shape());

        // Copy the input tensor if we can't mutate it.
        let tensor_original: TchTensor<E> =
            tensor.unary_ops(|tensor| tensor, |tensor| tensor.copy());
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

    pub fn gather(dim: usize, tensor: TchTensor<E>, indices: TchTensor<i64>) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.gather(dim as i64, &indices.tensor, false);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn scatter(
        dim: usize,
        tensor: TchTensor<E>,
        indices: TchTensor<i64>,
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let tensor = tensor
            .tensor
            .scatter_add(dim as i64, &indices.tensor, &value.tensor);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn index_select_dim(
        tensor: TchTensor<E>,
        dim: usize,
        indices: TchTensor<i64>,
    ) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.index_select(dim as i64, &indices.tensor);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn select_assign(
        tensor: TchTensor<E>,
        dim: usize,
        indices: TchTensor<i64>,
        value: TchTensor<E>,
    ) -> TchTensor<E> {
        tensor.clone().unary_ops(
            |mut tensor| tensor.index_add_(dim as i64, &indices.tensor, &value.tensor),
            |tensor| tensor.index_add(dim as i64, &indices.tensor, &value.tensor),
        )
    }

    pub fn cat(tensors: Vec<TchTensor<E>>, dim: usize) -> TchTensor<E> {
        let tensors: Vec<tch::Tensor> = tensors
            .into_iter()
            .map(|t| t.tensor.shallow_clone())
            .collect();
        let tensor = tch::Tensor::cat(&tensors, dim as i64);

        TchTensor::new(tensor)
    }

    pub fn equal(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.eq_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.eq_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.eq_tensor(rhs),
        )
    }

    pub fn equal_elem<S: Into<tch::Scalar> + Clone>(lhs: TchTensor<E>, rhs: S) -> TchTensor<bool> {
        lhs.unary_ops(
            |mut tensor| tensor.eq_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(rhs.clone().into()),
        )
    }

    pub fn greater(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.greater_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.less_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.greater_tensor(rhs),
        )
    }

    pub fn greater_elem<S: Into<tch::Scalar> + Clone>(
        lhs: TchTensor<E>,
        rhs: S,
    ) -> TchTensor<bool> {
        lhs.unary_ops(
            |mut tensor| tensor.greater_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.greater(rhs.clone().into()),
        )
    }

    pub fn greater_equal(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.greater_equal_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.less_equal_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.greater_equal_tensor(rhs),
        )
    }

    pub fn greater_equal_elem<S: Into<Scalar> + Clone>(
        lhs: TchTensor<E>,
        rhs: S,
    ) -> TchTensor<bool> {
        lhs.unary_ops(
            |mut tensor| {
                tensor
                    .greater_equal_(rhs.clone().into())
                    .to_kind(tch::Kind::Bool)
            },
            |tensor| tensor.greater_equal(rhs.clone().into()),
        )
    }

    pub fn lower(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.less_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.greater_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.less_tensor(rhs),
        )
    }

    pub fn lower_elem<S: Into<Scalar> + Clone>(lhs: TchTensor<E>, rhs: S) -> TchTensor<bool> {
        lhs.unary_ops(
            |mut tensor| tensor.less_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.less(rhs.clone().into()),
        )
    }

    pub fn lower_equal(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<bool> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.less_equal_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.greater_equal_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.less_equal_tensor(rhs),
        )
    }

    pub fn lower_equal_elem<S: Into<Scalar> + Clone>(lhs: TchTensor<E>, rhs: S) -> TchTensor<bool> {
        lhs.unary_ops(
            |mut tensor| {
                tensor
                    .less_equal_(rhs.clone().into())
                    .to_kind(tch::Kind::Bool)
            },
            |tensor| tensor.less_equal(rhs.clone().into()),
        )
    }

    pub fn add(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_add_(rhs).unwrap(),
            |lhs, rhs| rhs.f_add_(lhs).unwrap(),
            |lhs, rhs| lhs.f_add(rhs).unwrap(),
        )
    }

    pub fn sub(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_sub_(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
        )
    }

    pub fn mul(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_mul_(rhs).unwrap(),
            |lhs, rhs| rhs.f_mul_(lhs).unwrap(),
            |lhs, rhs| lhs.f_mul(rhs).unwrap(),
        )
    }

    pub fn div(lhs: TchTensor<E>, rhs: TchTensor<E>) -> TchTensor<E> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_div_(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
        )
    }

    pub fn mean(tensor: TchTensor<E>) -> TchTensor<E> {
        // view as 1d tensor
        let tensor = tensor.tensor.mean(E::KIND).view(1);
        TchTensor::new(tensor)
    }

    pub fn mean_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchTensor::from_existing(
            tensor
                .tensor
                .mean_dim(Some([dim as i64].as_slice()), true, E::KIND),
            tensor.storage,
        )
    }

    pub fn sum(tensor: TchTensor<E>) -> TchTensor<E> {
        // view as 1d tensor
        let tensor = tensor.tensor.sum(E::KIND).view(1);
        TchTensor::new(tensor)
    }

    pub fn sum_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchTensor::from_existing(
            tensor
                .tensor
                .sum_dim_intlist(Some([dim as i64].as_slice()), true, E::KIND),
            tensor.storage,
        )
    }

    pub fn prod(tensor: TchTensor<E>) -> TchTensor<E> {
        // view as 1d tensor
        let tensor = tensor.tensor.prod(E::KIND).view(1);
        TchTensor::new(tensor)
    }

    pub fn prod_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        TchTensor::from_existing(
            tensor.tensor.prod_dim_int(dim as i64, true, E::KIND),
            tensor.storage,
        )
    }

    pub fn argmax(tensor: TchTensor<E>, dim: usize) -> TchTensor<i64> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmax(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn argmin(tensor: TchTensor<E>, dim: usize) -> TchTensor<i64> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmin(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn max_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let (tensor, _indices) = tensor.tensor.max_dim(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn max_dim_with_indices(
        tensor: TchTensor<E>,
        dim: usize,
    ) -> (TchTensor<E>, TchTensor<i64>) {
        let storage = tensor.storage.clone();
        let (tensor, indices) = tensor.tensor.max_dim(dim as i64, true);

        let tensor = TchTensor::from_existing(tensor, storage);
        let indices = TchTensor::new(indices);

        (tensor, indices)
    }

    pub fn min_dim(tensor: TchTensor<E>, dim: usize) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let (tensor, _indices) = tensor.tensor.min_dim(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn min_dim_with_indices(
        tensor: TchTensor<E>,
        dim: usize,
    ) -> (TchTensor<E>, TchTensor<i64>) {
        let storage = tensor.storage.clone();
        let (tensor, indices) = tensor.tensor.min_dim(dim as i64, true);

        let tensor = TchTensor::from_existing(tensor, storage);
        let indices = TchTensor::new(indices);

        (tensor, indices)
    }

    pub fn clamp_min<S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor<E>,
        min: S,
    ) -> TchTensor<E> {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_min_(min),
            |tensor| tensor.clamp_min(min),
        )
    }

    pub fn clamp_max<S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor<E>,
        max: S,
    ) -> TchTensor<E> {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_max_(max),
            |tensor| tensor.clamp_max(max),
        )
    }

    pub fn clamp<S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor<E>,
        min: S,
        max: S,
    ) -> TchTensor<E> {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_(min, max),
            |tensor| tensor.clamp(min, max),
        )
    }

    pub fn swap_dims(tensor: TchTensor<E>, dim1: usize, dim2: usize) -> TchTensor<E> {
        let tensor = tensor.tensor.transpose(dim1 as i64, dim2 as i64);
        TchTensor::new(tensor)
    }

    pub fn permute(tensor: TchTensor<E>, axes: &[usize]) -> TchTensor<E> {
        let tensor = tensor
            .tensor
            .permute(axes.iter().map(|x| *x as i64).collect::<Vec<_>>());
        TchTensor::new(tensor)
    }

    pub fn flip(tensor: TchTensor<E>, axes: &[usize]) -> TchTensor<E> {
        let dims = axes.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let tensor = tensor.tensor.flip(dims);
        TchTensor::new(tensor)
    }

    pub fn narrow(tensor: TchTensor<E>, dim: usize, start: usize, length: usize) -> TchTensor<E> {
        TchTensor::new(
            tensor
                .tensor
                .narrow(dim as i64, start as i64, length as i64),
        )
    }

    pub fn chunk(tensor: TchTensor<E>, chunks: usize, dim: usize) -> Vec<TchTensor<E>> {
        tensor
            .tensor
            .chunk(chunks as i64, dim as i64)
            .into_iter()
            .map(|tensor| TchTensor::new(tensor))
            .collect()
    }

    pub fn powf(tensor: TchTensor<E>, exponent: TchTensor<E>) -> TchTensor<E> {
        TchTensor::binary_ops_tensor(
            tensor,
            exponent,
            |lhs, rhs| lhs.f_pow_tensor_(rhs).unwrap(),
            |lhs, rhs| lhs.f_pow(rhs).unwrap(),
            |lhs, rhs| lhs.f_pow(rhs).unwrap(),
        )
    }

    pub fn sign(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.sign_(), |tensor| tensor.sign())
    }

    pub fn expand(tensor: TchTensor<E>, shape: Shape) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let broadcasted_tensor = tensor.tensor.broadcast_to(TchShape::from(shape).dims);
        TchTensor::from_existing(broadcasted_tensor, storage)
    }

    pub fn sort(tensor: TchTensor<E>, dim: usize, descending: bool) -> TchTensor<E> {
        TchTensor::new(tensor.tensor.sort(dim as i64, descending).0)
    }

    pub fn sort_with_indices(
        tensor: TchTensor<E>,
        dim: usize,
        descending: bool,
    ) -> (TchTensor<E>, TchTensor<i64>) {
        let sorted = tensor.tensor.sort(dim as i64, descending);
        (TchTensor::new(sorted.0), TchTensor::new(sorted.1))
    }

    pub fn argsort(tensor: TchTensor<E>, dim: usize, descending: bool) -> TchTensor<i64> {
        TchTensor::new(tensor.tensor.argsort(dim as i64, descending))
    }

    pub fn quantize<I: tch::kind::Element>(
        tensor: TchTensor<E>,
        strategy: &QuantizationStrategy,
    ) -> TchTensor<I> {
        let mut tensor = tensor;
        // Quantize only works on Float Tensor
        if tensor.tensor.kind() == tch::Kind::Half {
            tensor.tensor = tensor.tensor.to_kind(tch::Kind::Float);
        }

        match strategy {
            QuantizationStrategy::PerTensorAffineInt8(ref q) => {
                TchTensor::new(tensor.tensor.quantize_per_tensor(
                    q.scale.into(),
                    q.offset.into(),
                    tch::Kind::QInt8,
                ))
            }
            QuantizationStrategy::PerTensorSymmetricInt8(ref q) => TchTensor::new(
                tensor
                    .tensor
                    .quantize_per_tensor(q.scale.into(), 0, tch::Kind::QInt8),
            ),
        }
    }
}
