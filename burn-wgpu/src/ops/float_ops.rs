use super::numeric::NumericOps;
use super::{BaseOps, Device, FloatElem, FloatTensor};
use crate::kernel::{unary, unary_inplace, unary_scalar, unary_scalar_inplace};
use crate::{
    element::{FloatElement, IntElement},
    unary, unary_inplace, GraphicsAPI, WGPUBackend, SEED,
};
use crate::{unary_scalar, unary_scalar_inplace};
use burn_common::rand::get_seeded_rng;
use burn_tensor::ElementConversion;
use burn_tensor::{backend::Backend, ops::TensorOps, Data, Distribution, Shape};

impl<G, F, I> TensorOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
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
        BaseOps::<G>::to_data(tensor)
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
        NumericOps::add(lhs, rhs)
    }

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::add_scalar(lhs, rhs)
    }

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::sub(lhs, rhs)
    }

    fn sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::sub_scalar(lhs, rhs)
    }

    fn mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::mul(lhs, rhs)
    }

    fn mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::mul_scalar(lhs, rhs)
    }

    fn div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        NumericOps::div(lhs, rhs)
    }

    fn div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        NumericOps::div_scalar(lhs, rhs)
    }

    fn matmul<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _dim1: usize,
        _dim2: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        _shape: Shape<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D2> {
        todo!()
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
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        _indexes: [std::ops::Range<usize>; D2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn index_assign<const D1: usize, const D2: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        _indexes: [std::ops::Range<usize>; D2],
        _value: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn mask_scatter<const D: usize>(
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
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal_elem<const D: usize>(
        _lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn sum<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn sum_dim<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mean<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn mean_dim<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
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

    fn log1p<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn powf<const D: usize>(lhs: FloatTensor<Self, D>, rhs: f32) -> FloatTensor<Self, D> {
        unary_scalar!(Powf, func "pow");
        unary_scalar_inplace!(PowfInplace, func "pow");

        if lhs.can_mut() {
            return unary_scalar_inplace::<PowfInplace, F, D>(lhs, rhs.elem());
        }

        unary_scalar::<Powf, F, D>(lhs, rhs.elem())
    }

    fn sqrt<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cos<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sin<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn tanh<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn erf<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cat<const D: usize>(
        _tensors: Vec<<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn argmax<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn argmin<const D: usize>(
        _tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        _dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
