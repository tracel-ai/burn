use super::numeric::NumericOps;
use super::{BaseOps, BoolTensor, Device, FloatElem, FloatTensor, IntTensor};
use crate::kernel::{
    self, matmul_tiling_2d_default, unary_default, unary_inplace_default, unary_scalar_default,
    unary_scalar_inplace_default,
};
use crate::unary_scalar_inplace;
use crate::{
    element::{FloatElement, IntElement},
    unary, unary_inplace, unary_scalar, GraphicsApi, WgpuBackend, SEED,
};
use burn_common::rand::get_seeded_rng;
use burn_tensor::ElementConversion;
use burn_tensor::{backend::Backend, ops::TensorOps, Data, Distribution, Shape};
use std::ops::Range;

impl<G, F, I> TensorOps<WgpuBackend<G, F, I>> for WgpuBackend<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn from_data<const D: usize>(
        data: Data<FloatElem<Self>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        BaseOps::<G>::from_data(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<FloatElem<Self>>,
        device: &Device<Self>,
    ) -> <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };
        let tensor = Self::from_data(Data::random(shape, distribution, &mut rng), device);
        *seed = Some(rng);
        tensor
    }

    fn shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn to_data<const D: usize>(tensor: &FloatTensor<Self, D>) -> Data<FloatElem<Self>, D> {
        BaseOps::<G>::into_data(tensor.clone())
    }

    fn into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> Data<FloatElem<Self>, D> {
        BaseOps::<G>::into_data(tensor)
    }

    fn device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.context.device.clone()
    }

    fn to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        BaseOps::<G>::to_device(tensor, device)
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        BaseOps::<G>::empty(shape, device)
    }

    fn add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::add(lhs, rhs)
    }

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::add_scalar(lhs, rhs)
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        NumericOps::<G>::zeros(shape, device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        NumericOps::<G>::ones(shape, device)
    }

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::sub(lhs, rhs)
    }

    fn sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::sub_scalar(lhs, rhs)
    }

    fn mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::mul(lhs, rhs)
    }

    fn mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::mul_scalar(lhs, rhs)
    }

    fn div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::div(lhs, rhs)
    }

    fn div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::<G>::div_scalar(lhs, rhs)
    }

    fn matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        let lhs = kernel::into_continuous(lhs);
        let rhs = kernel::into_continuous(rhs);

        matmul_tiling_2d_default::<FloatElem<Self>, D>(lhs, rhs)
    }

    fn swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        BaseOps::<G>::swap_dims(tensor, dim1, dim2)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        BaseOps::<G>::reshape(tensor, shape)
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::gather(dim, tensor, indices)
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::scatter(dim, tensor, indices, value)
    }

    fn select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        kernel::select(tensor, dim, indices)
    }

    fn select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::select_assign(tensor, dim, indices, value)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        BaseOps::<G>::mask_where(tensor, mask, value)
    }

    fn mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        BaseOps::<G>::mask_fill(tensor, mask, value)
    }

    fn equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::equal(lhs, rhs)
    }

    fn equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::equal_elem(lhs, rhs)
    }

    fn greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::greater(lhs, rhs)
    }

    fn greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::greater_elem(lhs, rhs)
    }

    fn greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::greater_equal(lhs, rhs)
    }

    fn greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::greater_equal_elem(lhs, rhs)
    }

    fn lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::lower(lhs, rhs)
    }

    fn lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::lower_elem(lhs, rhs)
    }

    fn lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::lower_equal(lhs, rhs)
    }

    fn lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        BaseOps::<G>::lower_equal_elem(lhs, rhs)
    }

    fn sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        NumericOps::<G>::sum(tensor)
    }

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        NumericOps::<G>::sum_dim(tensor, dim)
    }

    fn mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        NumericOps::<G>::mean(tensor)
    }

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        NumericOps::<G>::mean_dim(tensor, dim)
    }

    fn to_full_precision<const D: usize>(
        _tensor: &<WgpuBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <<WgpuBackend<G, F, I> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>
    {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        _tensor: <<WgpuBackend<G, F, I> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
    ) -> <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn exp<const D: usize>(lhs: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Exp, func "exp");
        unary_inplace!(ExpInplace, func "exp");

        if lhs.can_mut() {
            return unary_inplace_default::<ExpInplace, F, D>(lhs);
        }

        unary_default::<Exp, F, D>(lhs)
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Log, func "log");
        unary_inplace!(LogInplace, func "log");

        if tensor.can_mut() {
            return unary_inplace_default::<LogInplace, F, D>(tensor);
        }

        unary_default::<Log, F, D>(tensor)
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Log1p, body "output[id] = log(1.0 + input[id]);");
        unary_inplace!(Log1pInplace, body "input[id] = log(1.0 + input[id]);");

        if tensor.can_mut() {
            return unary_inplace_default::<Log1pInplace, F, D>(tensor);
        }

        unary_default::<Log1p, F, D>(tensor)
    }

    fn powf<const D: usize>(lhs: FloatTensor<Self, D>, rhs: f32) -> FloatTensor<Self, D> {
        unary_scalar!(Powf, func "powf", include "../template/powf.wgsl");
        unary_scalar_inplace!(PowfInplace, func "powf", include "../template/powf.wgsl");

        if lhs.can_mut() {
            return unary_scalar_inplace_default::<PowfInplace, F, D>(lhs, rhs.elem());
        }

        unary_scalar_default::<Powf, F, D>(lhs, rhs.elem())
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Sqrt, func "sqrt");
        unary_inplace!(SqrtInplace, func "sqrt");

        if tensor.can_mut() {
            return unary_inplace_default::<SqrtInplace, F, D>(tensor);
        }

        unary_default::<Sqrt, F, D>(tensor)
    }

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Cos, func "cos");
        unary_inplace!(CosInplace, func "cos");

        if tensor.can_mut() {
            return unary_inplace_default::<CosInplace, F, D>(tensor);
        }

        unary_default::<Cos, F, D>(tensor)
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Sin, func "sin");
        unary_inplace!(SinInplace, func "sin");

        if tensor.can_mut() {
            return unary_inplace_default::<SinInplace, F, D>(tensor);
        }

        unary_default::<Sin, F, D>(tensor)
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Tanh, func "tanh");
        unary_inplace!(TanhInplace, func "tanh");

        if tensor.can_mut() {
            return unary_inplace_default::<TanhInplace, F, D>(tensor);
        }

        unary_default::<Tanh, F, D>(tensor)
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Erf, func "erf", include "../template/erf.wgsl");
        unary_inplace!(ErfInplace, func "erf", include "../template/erf.wgsl");

        if tensor.can_mut() {
            return unary_inplace_default::<ErfInplace, F, D>(tensor);
        }

        unary_default::<Erf, F, D>(tensor)
    }

    fn cat<const D: usize>(tensors: Vec<FloatTensor<Self, D>>, dim: usize) -> FloatTensor<Self, D> {
        BaseOps::<G>::cat(tensors, dim)
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        NumericOps::<G>::argmax(tensor, dim)
    }

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        NumericOps::<G>::argmin(tensor, dim)
    }
}
