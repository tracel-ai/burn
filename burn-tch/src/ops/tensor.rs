use super::TchOps;
use crate::{element::TchElement, TchBackend, TchDevice, TchShape, TchTensor};
use burn_tensor::{backend::Backend, ops::TensorOps, Data, Distribution, ElementConversion, Shape};
use std::ops::Range;

impl<E: TchElement> TensorOps<TchBackend<E>> for TchBackend<E> {
    fn from_data<const D: usize>(data: Data<E, D>, device: &TchDevice) -> TchTensor<E, D> {
        TchTensor::from_data(data, (*device).into())
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<E>,
        device: &TchDevice,
    ) -> TchTensor<E, D> {
        match distribution {
            Distribution::Standard => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(0.0, 1.0)).unwrap()
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.f_bernoulli_float_(prob).unwrap())
                    .unwrap()
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor
                    .mut_ops(|tensor| tensor.uniform_(from.to_f64().unwrap(), to.to_f64().unwrap()))
                    .unwrap()
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, *device);
                tensor.mut_ops(|tensor| tensor.normal_(mean, std)).unwrap()
            }
        }
    }

    fn arange(range: Range<usize>, device: &TchDevice) -> TchTensor<i64, 1> {
        let device: tch::Device = (*device).into();
        let mut tensor = tch::Tensor::arange(range.end as i64, (tch::Kind::Int64, device));

        if range.start != 0 {
            tensor = tensor.f_add_scalar_(range.start as i64).unwrap();
        }

        TchTensor::new(tensor)
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &TchDevice) -> TchTensor<E, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(&shape.dims, (E::KIND, device)))
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &TchDevice) -> TchTensor<E, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(&shape.dims, (E::KIND, device)))
    }

    fn shape<const D: usize>(tensor: &<TchBackend<E> as Backend>::TensorPrimitive<D>) -> Shape<D> {
        tensor.shape()
    }

    fn to_data<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<TchBackend<E> as Backend>::FloatElem, D> {
        let values: Vec<E> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape())
    }

    fn into_data<const D: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<TchBackend<E> as Backend>::FloatElem, D> {
        let shape = tensor.shape();
        Data::new(tensor.tensor.into(), shape)
    }

    fn device<const D: usize>(tensor: &TchTensor<E, D>) -> TchDevice {
        tensor.tensor.device().into()
    }

    fn to_device<const D: usize>(tensor: TchTensor<E, D>, device: &TchDevice) -> TchTensor<E, D> {
        TchTensor::new(tensor.tensor.to((*device).into()))
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        let tensor = tch::Tensor::empty(&shape.dims.map(|a| a as i64), (E::KIND, (*device).into()));

        TchTensor::new(tensor)
    }

    fn add<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::add(lhs, rhs)
    }

    fn add_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        )
    }

    fn sub<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::sub(lhs, rhs)
    }

    fn sub_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        )
    }

    fn mul<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::mul(lhs, rhs)
    }

    fn mul_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        )
    }

    fn div<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        TchOps::div(lhs, rhs)
    }

    fn div_scalar<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<E, D> {
        let rhs: f64 = rhs.elem();

        lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        )
    }

    fn matmul<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = lhs.tensor.matmul(&rhs.tensor);
        TchTensor::new(tensor)
    }

    fn neg<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        Self::mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn swap_dims<const D: usize>(
        tensor: TchTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> TchTensor<E, D> {
        let tensor = tensor.tensor.transpose(dim1 as i64, dim2 as i64);
        TchTensor::new(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<E, D2> {
        TchOps::reshape(tensor, shape)
    }

    fn index_select<const D: usize>(
        tensor: TchTensor<E, D>,
        indexes: TchTensor<i64, D>,
    ) -> TchTensor<E, D> {
        TchOps::index_select(tensor, indexes)
    }

    fn index_select_assign<const D: usize>(
        tensor: TchTensor<E, D>,
        indexes: TchTensor<i64, D>,
        value: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        TchOps::index_select_assign(tensor, indexes, value)
    }

    fn index_select_dim<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        indexes: TchTensor<i64, 1>,
    ) -> TchTensor<E, D> {
        TchOps::index_select_dim(tensor, dim, indexes)
    }

    fn index_select_dim_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        dim: usize,
        indexes: TchTensor<i64, 1>,
        value: TchTensor<E, D2>,
    ) -> TchTensor<E, D1> {
        TchOps::index_select_dim_assign(tensor, dim, indexes, value)
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> TchTensor<E, D1> {
        TchOps::index(tensor, indexes)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: TchTensor<E, D1>,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D1> {
        TchOps::index_assign(tensor, indexes, value)
    }

    fn mask_scatter<const D: usize>(
        tensor: TchTensor<E, D>,
        mask: TchTensor<bool, D>,
        source: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        TchTensor::binary_ops_tensor(
            tensor,
            source,
            |tensor, source| tensor.f_masked_scatter_(&mask.tensor, source).unwrap(),
            |tensor, source| tensor.f_masked_scatter(&mask.tensor, source).unwrap(),
            |tensor, source| tensor.f_masked_scatter(&mask.tensor, source).unwrap(),
        )
    }

    fn mask_fill<const D: usize>(
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

    fn equal<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
        TchOps::equal(lhs, rhs)
    }

    fn equal_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::equal_elem(lhs, rhs.elem::<f64>())
    }

    fn greater<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
        TchOps::greater(lhs, rhs)
    }

    fn greater_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::greater_elem(lhs, rhs.elem::<f64>())
    }

    fn greater_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal(lhs, rhs)
    }

    fn greater_equal_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::greater_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn lower<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
        TchOps::lower(lhs, rhs)
    }

    fn lower_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::lower_elem(lhs, rhs.elem::<f64>())
    }

    fn lower_equal<const D: usize>(
        lhs: TchTensor<E, D>,
        rhs: TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower_equal(lhs, rhs)
    }

    fn lower_equal_elem<const D: usize>(lhs: TchTensor<E, D>, rhs: E) -> TchTensor<bool, D> {
        TchOps::lower_equal_elem(lhs, rhs.elem::<f64>())
    }

    fn mean<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        TchOps::mean(tensor)
    }

    fn sum<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        TchOps::sum(tensor)
    }

    fn mean_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::mean_dim(tensor, dim)
    }

    fn sum_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        TchOps::sum_dim(tensor, dim)
    }

    fn to_full_precision<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<f32, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(tch::Kind::Float);

        TchTensor::from_existing(tensor, storage)
    }

    fn from_full_precision<const D: usize>(tensor: TchTensor<f32, D>) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.to_kind(E::KIND);

        TchTensor::from_existing(tensor, storage)
    }

    fn argmax<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmax(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    fn argmin<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.argmin(dim as i64, true);

        TchTensor::from_existing(tensor, storage)
    }

    fn exp<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.exp_(), |tensor| tensor.exp())
    }

    fn log<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.log_(), |tensor| tensor.log())
    }

    fn log1p<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.log1p_(), |tensor| tensor.log1p())
    }

    fn powf<const D: usize>(tensor: TchTensor<E, D>, value: f32) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.f_pow_(value as f64).unwrap(),
            |tensor| tensor.pow_tensor_scalar(value as f64),
        )
    }

    fn sqrt<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.sqrt_(), |tensor| tensor.sqrt())
    }

    fn cos<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.cos_(), |tensor| tensor.cos())
    }

    fn sin<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.sin_(), |tensor| tensor.sin())
    }

    fn tanh<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.tanh_(), |tensor| tensor.tanh())
    }

    fn erf<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.erf_(), |tensor| tensor.erf())
    }

    fn cat<const D: usize>(tensors: Vec<TchTensor<E, D>>, dim: usize) -> TchTensor<E, D> {
        TchOps::cat(tensors, dim)
    }

    fn relu<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.relu_(), |tensor| tensor.relu())
    }
    fn unbind<const D: usize, const D2: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
    ) -> Vec<TchTensor<E, D2>> {
        TchOps::unbind(tensor, dim)
    }
    fn cumsum<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.cumsum_(dim as i64, E::KIND),
            |tensor| tensor.cumsum(dim as i64, E::KIND),
        )
    }
    fn stack<const D: usize, const D2: usize>(
        tensors: Vec<TchTensor<E, D>>,
        dim: usize,
    ) -> TchTensor<E, D2> {
        TchOps::stack(tensors, dim)
    }
    fn narrow<const D: usize>(
        tensor: TchTensor<E, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> TchTensor<E, D> {
        tensor.unary_ops(
            |tensor| tensor.narrow(dim as i64, start as i64, length as i64),
            |tensor| tensor.narrow(dim as i64, start as i64, length as i64),
        )
    }
    fn upsample_linear1d<const D: usize, const D2: usize>(
        tensor: TchTensor<E, D>,
        output_size: &[usize],
        align_corners: bool,
        scales: impl Into<Option<f64>>,
    ) -> TchTensor<E, D2> {
        let out_i64 = output_size.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let scales = scales.into();
        tensor.unary_ops(
            |tensor| tensor.upsample_linear1d(&out_i64, align_corners, scales),
            |tensor| tensor.upsample_linear1d(&out_i64, align_corners, scales),
        )
    }
    fn pad<const D: usize>(
        tensor: TchTensor<E, D>,
        pad: &[usize],
        mode: &str,
        value: impl Into<Option<f64>>,
    ) -> TchTensor<E, D> {
        let pad = pad.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let value = value.into();
        tensor.unary_ops(
            |tensor| tensor.pad(&pad, mode, value),
            |tensor| tensor.pad(&pad, mode, value),
        )
    }
    fn expand<const D: usize>(
        tensor: TchTensor<E, D>,
        size: Vec<usize>,
        implicit: bool,
    ) -> TchTensor<E, D> {
        let size = size
            .iter()
            .map(|x| match x {
                &usize::MAX => -1_i64,
                _ => *x as i64,
            })
            .collect::<Vec<_>>();
        tensor.unary_ops(
            |tensor| tensor.expand(&size, implicit),
            |tensor| tensor.expand(&size, implicit),
        )
    }
    fn upsample_bilinear2d<const D: usize, const D2: usize>(
        tensor: TchTensor<E, D>,
        output_size: Vec<usize>,
        align_corners: bool,
        scales_h: impl Into<Option<f64>>,
        scales_w: impl Into<Option<f64>>,
    ) -> TchTensor<E, D2> {
        let output_size = output_size.iter().map(|x| *x as i64).collect::<Vec<_>>();
        let scales_h = scales_h.into();
        let scales_w = scales_w.into();
        tensor.unary_ops(
            |tensor| tensor.upsample_bilinear2d(&output_size, align_corners, scales_h, scales_w),
            |tensor| tensor.upsample_bilinear2d(&output_size, align_corners, scales_h, scales_w),
        )
    }
    fn select<const D: usize, const D2: usize>(
        tensor: TchTensor<E, D>,
        dim: i64,
        index: i64,
    ) -> TchTensor<E, D2> {
        tensor.unary_ops(
            |tensor| tensor.select(dim as i64, index as i64),
            |tensor| tensor.select(dim as i64, index as i64),
        )
    }
    fn flip<const D: usize>(tensor: TchTensor<E, D>, dims: Vec<usize>) -> TchTensor<E, D> {
        let dims = dims.iter().map(|x| *x as i64).collect::<Vec<_>>();
        tensor.unary_ops(|tensor| tensor.flip(&dims), |tensor| tensor.flip(&dims))
    }
    fn permute<const D: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
        dims: [usize; D],
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        let dims = dims.iter().map(|x| *x as i64).collect::<Vec<_>>();
        tensor.unary_ops(
            |tensor| tensor.permute(&dims),
            |tensor| tensor.permute(&dims),
        )
    }
    fn einsum<const D: usize, const D2: usize, const D3: usize>(
        equation: &str,
        tensor1: <TchBackend<E> as Backend>::TensorPrimitive<D>,
        tensor2: <TchBackend<E> as Backend>::TensorPrimitive<D2>,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D3> {
        let res = tch::Tensor::einsum(equation, &[tensor1.tensor, tensor2.tensor], None);
        TchTensor::new(res)
    }
    fn index_tch<const D: usize, const D2: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
        indices: Vec<<TchBackend<E> as Backend>::IntTensorPrimitive<D>>,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D2> {
        let indices: Vec<_> = indices.iter().map(|x| Some(x.tensor.copy())).collect();
        tensor.unary_ops(
            |tensor| tensor.index(&indices),
            |tensor| tensor.index(&indices),
        )
    }
    fn repeat_interleave_self_int<const D: usize, const D2: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
        repeats: usize,
        dim: Option<usize>,
        output_size: Option<usize>,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D2> {
        let dim = match dim {
            Some(x) => Some(x as i64),
            None => None,
        };
        let output_size = output_size.map(|x| x as i64);
        tensor.unary_ops(
            |tensor| tensor.repeat_interleave_self_int(repeats as i64, dim, output_size),
            |tensor| tensor.repeat_interleave_self_int(repeats as i64, dim, output_size),
        )
    }
    fn where_self<const D: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
        condition: <TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
        other: <TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        tensor.unary_ops(
            |tensor| tensor.where_self(&condition.tensor, &other.tensor),
            |tensor| tensor.where_self(&condition.tensor, &other.tensor),
        )
    }
}
