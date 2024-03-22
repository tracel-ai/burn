use burn_tensor::Shape;
use tch::Scalar;

use crate::{LibTorchDevice, TchShape, TchTensor};
use std::{marker::PhantomData, ops::Range};

pub struct TchOps<E: tch::kind::Element + Copy + Default> {
    e: PhantomData<E>,
}

impl<E: tch::kind::Element + Copy + Default> TchOps<E> {
    pub fn to_device<const D: usize>(
        tensor: TchTensor<E, D>,
        device: &LibTorchDevice,
    ) -> TchTensor<E, D> {
        let device = (*device).into();

        // We have to manually check if the device is the same, since when it's the case, we need to keep
        // the same storage reference and not create a new one.
        if tensor.tensor.device() == device {
            return tensor;
        }

        TchTensor::new(tensor.tensor.to(device))
    }

    pub fn reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<E, D2> {
        let shape_tch: TchShape<D2> = shape.into();

        TchTensor::from_existing(tensor.tensor.reshape(shape_tch.dims), tensor.storage)
    }

    pub fn repeat<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        times: usize,
    ) -> TchTensor<E, D> {
        let mut dims = [1; D];
        dims[dim] = times as i64;
        let tensor = tch::Tensor::repeat(&tensor.tensor, dims);
        TchTensor::new(tensor)
    }

    pub fn slice<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        ranges: [Range<usize>; D2],
    ) -> TchTensor<E, D1> {
        let storage = tensor.storage.clone();
        let mut tensor = tensor.tensor.shallow_clone();

        for (i, index) in ranges.iter().enumerate().take(D2) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;
            tensor = tensor.narrow(i as i64, start, length);
        }

        TchTensor::partial(tensor, storage)
    }

    pub fn slice_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        ranges: [Range<usize>; D2],
        value: TchTensor<E, D1>,
    ) -> TchTensor<E, D1> {
        let tch_shape = TchShape::from(tensor.shape());

        // Copy the input tensor if we can't mutate it.
        let tensor_original: TchTensor<E, D1> =
            tensor.unary_ops(|tensor| tensor, |tensor| tensor.copy());
        let tensor_original = tensor_original.tensor;

        let mut tensor = tensor_original.view_(tch_shape.dims);

        for (i, index) in ranges.into_iter().enumerate().take(D2) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;

            tensor = tensor.narrow(i as i64, start, length);
        }

        tensor.copy_(&value.tensor);

        TchTensor::new(tensor_original)
    }

    pub fn gather<const D: usize>(
        dim: usize,
        tensor: TchTensor<E, D>,
        indices: TchTensor<i64, D>,
    ) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.gather(dim as i64, &indices.tensor, false);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn scatter<const D: usize>(
        dim: usize,
        tensor: TchTensor<E, D>,
        indices: TchTensor<i64, D>,
        value: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor
            .tensor
            .scatter_add(dim as i64, &indices.tensor, &value.tensor);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn index_select_dim<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        indices: TchTensor<i64, 1>,
    ) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.index_select(dim as i64, &indices.tensor);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn select_assign<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        indices_tensor: TchTensor<i64, 1>,
        value: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        let mut indices = Vec::with_capacity(D);
        for _ in 0..D {
            indices.push(None);
        }
        indices[dim] = Some(indices_tensor.tensor);

        tensor.unary_ops(
            |mut tensor| tensor.index_put_(&indices, &value.tensor, true),
            |tensor| tensor.index_put(&indices, &value.tensor, true),
        )
    }

    pub fn cat<const D: usize>(tensors: Vec<TchTensor<E, D>>, dim: usize) -> TchTensor<E, D> {
        let tensors: Vec<tch::Tensor> = tensors
            .into_iter()
            .map(|t| t.tensor.shallow_clone())
            .collect();
        let tensor = tch::Tensor::cat(&tensors, dim as i64);

        TchTensor::new(tensor)
    }

    pub fn equal<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.eq_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.eq_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.eq_tensor(rhs),
        )
    }

    pub fn equal_elem<const D: usize, S: Into<tch::Scalar> + Clone>(
        lhs: TchTensor<E, D>,
        rhs: S,
    ) -> TchTensor<bool, D> {
        lhs.unary_ops(
            |mut tensor| tensor.eq_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(rhs.clone().into()),
        )
    }

    pub fn greater<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.greater_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.less_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.greater_tensor(rhs),
        )
    }

    pub fn greater_elem<const D: usize, S: Into<tch::Scalar> + Clone>(
        lhs: TchTensor<E, D>,
        rhs: S,
    ) -> TchTensor<bool, D> {
        lhs.unary_ops(
            |mut tensor| tensor.greater_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.greater(rhs.clone().into()),
        )
    }

    pub fn greater_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.greater_equal_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.less_equal_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.greater_equal_tensor(rhs),
        )
    }

    pub fn greater_equal_elem<const D: usize, S: Into<Scalar> + Clone>(
        lhs: TchTensor<E, D>,
        rhs: S,
    ) -> TchTensor<bool, D> {
        lhs.unary_ops(
            |mut tensor| {
                tensor
                    .greater_equal_(rhs.clone().into())
                    .to_kind(tch::Kind::Bool)
            },
            |tensor| tensor.greater_equal(rhs.clone().into()),
        )
    }

    pub fn lower<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.less_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.greater_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.less_tensor(rhs),
        )
    }

    pub fn lower_elem<const D: usize, S: Into<Scalar> + Clone>(
        lhs: TchTensor<E, D>,
        rhs: S,
    ) -> TchTensor<bool, D> {
        lhs.unary_ops(
            |mut tensor| tensor.less_(rhs.clone().into()).to_kind(tch::Kind::Bool),
            |tensor| tensor.less(rhs.clone().into()),
        )
    }

    pub fn lower_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.less_equal_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.greater_equal_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.less_equal_tensor(rhs),
        )
    }

    pub fn lower_equal_elem<const D: usize, S: Into<Scalar> + Clone>(
        lhs: TchTensor<E, D>,
        rhs: S,
    ) -> TchTensor<bool, D> {
        lhs.unary_ops(
            |mut tensor| {
                tensor
                    .less_equal_(rhs.clone().into())
                    .to_kind(tch::Kind::Bool)
            },
            |tensor| tensor.less_equal(rhs.clone().into()),
        )
    }

    pub fn add<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_add_(rhs).unwrap(),
            |lhs, rhs| rhs.f_add_(lhs).unwrap(),
            |lhs, rhs| lhs.f_add(rhs).unwrap(),
        )
    }

    pub fn sub<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_sub_(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
        )
    }

    pub fn mul<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_mul_(rhs).unwrap(),
            |lhs, rhs| rhs.f_mul_(lhs).unwrap(),
            |lhs, rhs| lhs.f_mul(rhs).unwrap(),
        )
    }

    pub fn div<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_div_(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
        )
    }

    pub fn mean<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.mean(E::KIND);
        TchTensor::new(tensor)
    }

    pub fn mean_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchTensor::from_existing(
            tensor
                .tensor
                .mean_dim(Some([dim as i64].as_slice()), true, E::KIND),
            tensor.storage,
        )
    }

    pub fn sum<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.sum(E::KIND);
        TchTensor::new(tensor)
    }

    pub fn sum_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchTensor::from_existing(
            tensor
                .tensor
                .sum_dim_intlist(Some([dim as i64].as_slice()), true, E::KIND),
            tensor.storage,
        )
    }

    pub fn prod<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.prod(E::KIND);
        TchTensor::new(tensor)
    }

    pub fn prod_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchTensor::from_existing(
            tensor.tensor.prod_dim_int(dim as i64, true, E::KIND),
            tensor.storage,
        )
    }

    pub fn argmax<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmax(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn argmin<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmin(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn max_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let (tensor, _indices) = tensor.tensor.max_dim(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn max_dim_with_indices<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
    ) -> (TchTensor<E, D>, TchTensor<i64, D>) {
        let storage = tensor.storage.clone();
        let (tensor, indices) = tensor.tensor.max_dim(dim as i64, true);

        let tensor = TchTensor::from_existing(tensor, storage);
        let indices = TchTensor::new(indices);

        (tensor, indices)
    }

    pub fn min_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let (tensor, _indices) = tensor.tensor.min_dim(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    pub fn min_dim_with_indices<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
    ) -> (TchTensor<E, D>, TchTensor<i64, D>) {
        let storage = tensor.storage.clone();
        let (tensor, indices) = tensor.tensor.min_dim(dim as i64, true);

        let tensor = TchTensor::from_existing(tensor, storage);
        let indices = TchTensor::new(indices);

        (tensor, indices)
    }

    pub fn clamp_min<const D: usize, S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor<E, D>,
        min: S,
    ) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_min_(min),
            |tensor| tensor.clamp_min(min),
        )
    }

    pub fn clamp_max<const D: usize, S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor<E, D>,
        max: S,
    ) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_max_(max),
            |tensor| tensor.clamp_max(max),
        )
    }

    pub fn clamp<const D: usize, S: Into<tch::Scalar> + Clone + Copy>(
        tensor: TchTensor<E, D>,
        min: S,
        max: S,
    ) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.clamp_(min, max),
            |tensor| tensor.clamp(min, max),
        )
    }

    pub fn swap_dims<const D: usize>(
        tensor: TchTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> TchTensor<E, D> {
        let tensor = tensor.tensor.transpose(dim1 as i64, dim2 as i64);
        TchTensor::new(tensor)
    }

    pub fn permute<const D: usize>(tensor: TchTensor<E, D>, axes: [usize; D]) -> TchTensor<E, D> {
        let tensor = tensor.tensor.permute(axes.map(|x| x as i64));
        TchTensor::new(tensor)
    }

    pub fn flip<const D: usize>(tensor: TchTensor<E, D>, axes: &[usize]) -> TchTensor<E, D> {
        let dims = axes.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let tensor = tensor.tensor.flip(dims);
        TchTensor::new(tensor)
    }

    pub fn narrow<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> TchTensor<E, D> {
        TchTensor::new(
            tensor
                .tensor
                .narrow(dim as i64, start as i64, length as i64),
        )
    }

    pub fn chunk<const D: usize>(
        tensor: TchTensor<E, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<TchTensor<E, D>> {
        tensor
            .tensor
            .chunk(chunks as i64, dim as i64)
            .into_iter()
            .map(|tensor| TchTensor::new(tensor))
            .collect()
    }

    pub fn powf<const D: usize>(
        tensor: TchTensor<E, D>,
        exponent: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        TchTensor::binary_ops_tensor(
            tensor,
            exponent,
            |lhs, rhs| lhs.f_pow_tensor_(rhs).unwrap(),
            |lhs, rhs| lhs.f_pow(rhs).unwrap(),
            |lhs, rhs| lhs.f_pow(rhs).unwrap(),
        )
    }

    pub fn sign<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.sign_(), |tensor| tensor.sign())
    }

    pub fn expand<const D: usize, const D2: usize>(
        tensor: TchTensor<E, D>,
        shape: Shape<D2>,
    ) -> TchTensor<E, D2> {
        let tensor = tensor.tensor.broadcast_to(shape.dims.map(|x| x as i64));
        TchTensor::new(tensor)
    }

    pub fn sort<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        descending: bool,
    ) -> TchTensor<E, D> {
        TchTensor::new(tensor.tensor.sort(dim as i64, descending).0)
    }

    pub fn argsort<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        descending: bool,
    ) -> TchTensor<i64, D> {
        TchTensor::new(tensor.tensor.argsort(dim as i64, descending))
    }
}
