use super::{numeric, BoolTensor, Device, FloatElem, FloatTensor, FullPrecisionBackend, IntTensor};
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
use crate::kernel::{
    self, unary_default, unary_inplace_default, unary_scalar_default, unary_scalar_inplace_default,
};

use crate::unary_scalar_inplace;
use crate::{
    element::{FloatElement, IntElement},
    unary, unary_inplace, unary_scalar, GraphicsApi, WgpuBackend,
};
use burn_tensor::ElementConversion;
use burn_tensor::{ops::TensorOps, Data, Distribution, Shape};

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
        super::from_data::<G, F, D>(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<FloatElem<Self>>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        match distribution {
            Distribution::Default => random_uniform::<G, F, D>(shape, device, 0.elem(), 1.elem()),
            Distribution::Uniform(low, high) => random_uniform::<G, F, D>(shape, device, low, high),
            Distribution::Bernoulli(prob) => {
                random_bernoulli::<G, F, D>(shape, device, prob.elem())
            }
            Distribution::Normal(mean, std) => {
                random_normal::<G, F, D>(shape, device, mean.elem(), std.elem())
            }
        }
    }

    fn shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn to_data<const D: usize>(tensor: &FloatTensor<Self, D>) -> Data<FloatElem<Self>, D> {
        super::into_data(tensor.clone())
    }

    fn into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> Data<FloatElem<Self>, D> {
        super::into_data(tensor)
    }

    fn device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.context.device.clone()
    }

    fn to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        super::to_device::<G, F, D>(tensor, device)
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        super::empty::<G, F, D>(shape, device)
    }

    fn add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::add(lhs, rhs)
    }

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::add_scalar(lhs, rhs)
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        numeric::zeros::<G, F, D>(shape, device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        numeric::ones::<G, F, D>(shape, device)
    }

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::sub(lhs, rhs)
    }

    fn sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::sub_scalar(lhs, rhs)
    }

    fn mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::mul(lhs, rhs)
    }

    fn mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::mul_scalar(lhs, rhs)
    }

    fn div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::div(lhs, rhs)
    }

    fn div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::div_scalar(lhs, rhs)
    }

    fn matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[cfg(feature = "autotune")]
        return kernel::matmul::tune::<G, F, D>(lhs, rhs);

        #[cfg(not(feature = "autotune"))]
        kernel::matmul::contiguous_vectorized::matmul_tiling_2d_default(lhs, rhs)
    }

    fn swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        super::swap_dims(tensor, dim1, dim2)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        super::reshape(tensor, shape)
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
        kernel::mask_where(tensor, mask, value)
    }

    fn mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        kernel::mask_fill(tensor, mask, value)
    }

    fn equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal(lhs, rhs)
    }

    fn equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::equal_elem(lhs, rhs)
    }

    fn greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater(lhs, rhs)
    }

    fn greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_elem(lhs, rhs)
    }

    fn greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal(lhs, rhs)
    }

    fn greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal_elem(lhs, rhs)
    }

    fn lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower(lhs, rhs)
    }

    fn lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_elem(lhs, rhs)
    }

    fn lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal(lhs, rhs)
    }

    fn lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal_elem(lhs, rhs)
    }

    fn sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        kernel::sum(tensor)
    }

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        kernel::sum_dim(tensor, dim)
    }

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        kernel::mean_dim(tensor, dim)
    }

    fn to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        kernel::cast(tensor.clone())
    }

    fn from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        kernel::cast(tensor)
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

    fn abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Abs, func "abs");
        unary_inplace!(AbsInplace, func "abs");

        if tensor.can_mut() {
            return unary_inplace_default::<AbsInplace, F, D>(tensor);
        }

        unary_default::<Abs, F, D>(tensor)
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
        // Metal has a weird numerical behaviour with tanh which require a new function
        #[cfg(target_os = "macos")]
        unary!(Tanh, func "safe_tanh", include "../template/safe_tanh.wgsl");
        #[cfg(target_os = "macos")]
        unary_inplace!(TanhInplace, func "safe_tanh", include "../template/safe_tanh.wgsl");

        #[cfg(not(target_os = "macos"))]
        unary!(Tanh, func "tanh");
        #[cfg(not(target_os = "macos"))]
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
        kernel::cat(tensors, dim)
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::argmax(tensor, dim)
    }

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::argmin(tensor, dim)
    }

    fn into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        kernel::cast(tensor)
    }

    // TODO implement clamp kernels (see https://github.com/burn-rs/burn/issues/549)
    // fn clamp_min<const D: usize>(
    //     tensor: FloatTensor<Self, D>,
    //     min: FloatElem<Self>,
    // ) -> FloatTensor<Self, D> {
    //     kernel::clamp_min(tensor, min)
    // }

    // fn clamp_max<const D: usize>(
    //     tensor: FloatTensor<Self, D>,
    //     max: FloatElem<Self>,
    // ) -> FloatTensor<Self, D> {
    //     kernel::clamp_max(tensor, max)
    // }

    // fn clamp<const D: usize>(
    //     tensor: FloatTensor<Self, D>,
    //     min: FloatElem<Self>,
    //     max: FloatElem<Self>,
    // ) -> FloatTensor<Self, D> {
    //     kernel::clamp(tensor, min, max)
    // }
}
