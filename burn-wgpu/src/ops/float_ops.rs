use super::numeric;
use crate::codegen::{Elem, Operator, Variable};
#[cfg(not(feature = "autotune"))]
use crate::kernel::matmul::init_matmul_output;
#[cfg(feature = "autotune")]
use crate::kernel::matmul::matmul_autotune;
#[cfg(not(feature = "autotune"))]
use crate::kernel::matmul::vec4::matmul_tiling_2d_vec4;
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
#[cfg(not(feature = "autotune"))]
use crate::kernel::reduce::init_reduce_output;
use crate::kernel::{self, reduce};
use crate::WgpuDevice;
use crate::{unary, FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_tensor::ops::{
    BoolTensor, Device, FloatElem, FloatTensor, FullPrecisionBackend, IntTensor,
};
use burn_tensor::{ops::TensorOps, Data, Distribution, Shape};
use burn_tensor::{ElementConversion, Reader};
use std::ops::Range;

impl<G, F, I> TensorOps<Wgpu<G, F, I>> for Wgpu<G, F, I>
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
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        match distribution {
            Distribution::Default => random_uniform::<G, F, D>(shape, device, 0.elem(), 1.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform::<G, F, D>(shape, device, low.elem(), high.elem())
            }
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

    fn into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> Reader<Data<FloatElem<Self>, D>> {
        super::into_data(tensor)
    }

    fn device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
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

    fn full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &WgpuDevice,
    ) -> FloatTensor<Self, D> {
        numeric::full::<G, F, D>(shape, device, fill_value)
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
        {
            matmul_autotune(lhs, rhs)
        }

        #[cfg(not(feature = "autotune"))]
        {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_vec4(lhs, rhs, out)
        }
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
        reduce::sum(tensor)
    }

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        #[cfg(feature = "autotune")]
        {
            reduce::sum_dim_autotune(tensor, dim)
        }

        #[cfg(not(feature = "autotune"))]
        {
            let output = init_reduce_output(&tensor, dim);
            reduce::sum_dim(tensor, output, dim)
        }
    }

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        #[cfg(feature = "autotune")]
        {
            reduce::mean_dim_autotune(tensor, dim)
        }

        #[cfg(not(feature = "autotune"))]
        {
            let output = init_reduce_output(&tensor, dim);
            reduce::mean_dim(tensor, output, dim)
        }
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

    fn exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Exp {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Log {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Log1p {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn powf<const D: usize>(lhs: FloatTensor<Self, D>, rhs: f32) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Powf {
                lhs: Variable::Input(0, elem),
                rhs: Variable::Scalar(0, elem),
                out: Variable::Local(0, elem),
            },
            input: lhs; rhs.elem(),
            elem: F
        )
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Sqrt {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Abs {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Cos {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Sin {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Tanh {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(
            operator: |elem: Elem| Operator::Erf {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn cat<const D: usize>(tensors: Vec<FloatTensor<Self, D>>, dim: usize) -> FloatTensor<Self, D> {
        kernel::cat(tensors, dim)
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        reduce::argmax(tensor, dim)
    }

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        reduce::argmin(tensor, dim)
    }

    fn into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        kernel::cast(tensor)
    }

    fn clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        kernel::clamp(tensor, min, max)
    }

    fn recip<const D: usize>(
        tensor: FloatTensor<Wgpu<G, F, I>, D>,
    ) -> FloatTensor<Wgpu<G, F, I>, D> {
        unary!(
            operator: |elem: Elem| Operator::Recip {
                input: Variable::Input(0, elem),
                out: Variable::Local(0, elem),
            },
            input: tensor,
            elem: F
        )
    }

    fn repeat<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<Self, D> {
        kernel::repeat(tensor, dim, times)
    }
}
