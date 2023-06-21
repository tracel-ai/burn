use super::numeric::NumericOps;
use super::{BaseOps, BoolTensor, Device, FloatElem, FloatTensor, IntTensor};
use crate::kernel::{matmul, unary, unary_inplace, unary_scalar, unary_scalar_inplace};
use crate::{
    element::{FloatElement, IntElement},
    unary, unary_inplace, GraphicsApi, WGPUBackend, SEED,
};
use crate::{unary_scalar, unary_scalar_inplace};
use burn_common::rand::get_seeded_rng;
use burn_tensor::ElementConversion;
use burn_tensor::{backend::Backend, ops::TensorOps, Data, Distribution, Shape};
use std::ops::Range;

impl<G, F, I> TensorOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
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
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
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
        let lhs = BaseOps::<G>::into_continuous(lhs);
        let rhs = BaseOps::<G>::into_continuous(rhs);

        matmul::<FloatElem<Self>, D>(lhs, rhs)
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
        _dim: usize,
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn scatter<const D: usize>(
        _dim: usize,
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        _value: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn index_select<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _dim: usize,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn index_select_assign<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        _dim: usize,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
        _value: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        indexes: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        BaseOps::<G>::index(tensor, indexes)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        indexes: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        BaseOps::<G>::index_assign(tensor, indexes, value)
    }

    fn mask_where<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _source: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        _value: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
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
        _tensor: &<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <<WGPUBackend<G, F, I> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>
    {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        _tensor: <<WGPUBackend<G, F, I> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn exp<const D: usize>(lhs: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Exp, func "exp");
        unary_inplace!(ExpInplace, func "exp");

        if lhs.can_mut() {
            return unary_inplace::<ExpInplace, F, D>(lhs);
        }

        unary::<Exp, F, D>(lhs)
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Log, func "log");
        unary_inplace!(LogInplace, func "log");

        if tensor.can_mut() {
            return unary_inplace::<LogInplace, F, D>(tensor);
        }

        unary::<Log, F, D>(tensor)
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Log1p, body "output[global_id.x] = log(1.0 + input[global_id.x]);");
        unary_inplace!(Log1pInplace, body "input[global_id.x] = log(1.0 + input[global_id.x]);");

        if tensor.can_mut() {
            return unary_inplace::<Log1pInplace, F, D>(tensor);
        }

        unary::<Log1p, F, D>(tensor)
    }

    fn powf<const D: usize>(lhs: FloatTensor<Self, D>, rhs: f32) -> FloatTensor<Self, D> {
        unary_scalar!(Powf, func "pow");
        unary_scalar_inplace!(PowfInplace, func "pow");

        if lhs.can_mut() {
            return unary_scalar_inplace::<PowfInplace, F, D>(lhs, rhs.elem());
        }

        unary_scalar::<Powf, F, D>(lhs, rhs.elem())
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Sqrt, func "sqrt");
        unary_inplace!(SqrtInplace, func "sqrt");

        if tensor.can_mut() {
            return unary_inplace::<SqrtInplace, F, D>(tensor);
        }

        unary::<Sqrt, F, D>(tensor)
    }

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Cos, func "cos");
        unary_inplace!(CosInplace, func "cos");

        if tensor.can_mut() {
            return unary_inplace::<CosInplace, F, D>(tensor);
        }

        unary::<Cos, F, D>(tensor)
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Sin, func "sin");
        unary_inplace!(SinInplace, func "sin");

        if tensor.can_mut() {
            return unary_inplace::<SinInplace, F, D>(tensor);
        }

        unary::<Sin, F, D>(tensor)
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Tanh, func "tanh");
        unary_inplace!(TanhInplace, func "tanh");

        if tensor.can_mut() {
            return unary_inplace::<TanhInplace, F, D>(tensor);
        }

        unary::<Tanh, F, D>(tensor)
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary!(Erf, func "erf", include "../template/erf.wgsl");
        unary_inplace!(ErfInplace, func "erf", include "../template/erf.wgsl");

        if tensor.can_mut() {
            return unary_inplace::<ErfInplace, F, D>(tensor);
        }

        unary::<Erf, F, D>(tensor)
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
