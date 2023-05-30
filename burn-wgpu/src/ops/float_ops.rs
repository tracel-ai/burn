use super::{Device, FloatElem, FloatTensor};
use crate::kernel_elemwise_inplace;
use crate::tensor::elemwise::{execute_elemwise, execute_elemwise_inplace};
use crate::{
    element::{FloatElement, IntElement},
    kernel::KernelTemplate,
    kernel_elemwise,
    pool::get_context,
    tensor::WGPUTensor,
    GraphicsAPI, WGPUBackend, SEED,
};
use burn_common::rand::get_seeded_rng;
use burn_tensor::{backend::Backend, ops::TensorOps, Data, Distribution, Shape};
use std::sync::Arc;

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
        let context = get_context::<G>(device);
        let buffer = context.create_buffer_with_data(bytemuck::cast_slice(&data.value));

        WGPUTensor::new(context, data.shape, Arc::new(buffer))
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
        let bytes = tensor.context.buffer_to_data(&tensor.buffer);
        let values = bytemuck::cast_slice(&bytes);

        Data::new(values.to_vec(), tensor.shape.clone())
    }

    fn into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> Data<FloatElem<Self>, D> {
        Self::to_data(&tensor)
    }

    fn device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.context.device_wgpu.clone()
    }

    fn to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        if &tensor.context.device_wgpu == device {
            return tensor;
        }

        let context = get_context::<G>(device);
        tensor.to_context(context)
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        let context = get_context::<G>(device);
        let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<F>());

        WGPUTensor::new(context, shape, Arc::new(buffer))
    }

    fn add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel_elemwise!(Add, "+");
        kernel_elemwise_inplace!(AddInplace, "+");

        if lhs.can_mut_broadcast(&rhs) {
            return execute_elemwise_inplace::<AddInplace, F, D>(lhs, rhs);
        }

        if rhs.can_mut_broadcast(&lhs) {
            return execute_elemwise_inplace::<AddInplace, F, D>(rhs, lhs);
        }

        execute_elemwise::<Add, F, D>(lhs, rhs)
    }

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel_elemwise!(Sub, "-");
        kernel_elemwise_inplace!(SubInplace, "-");

        if lhs.can_mut_broadcast(&rhs) {
            return execute_elemwise_inplace::<SubInplace, F, D>(lhs, rhs);
        }

        execute_elemwise::<Sub, F, D>(lhs, rhs)
    }

    fn sub_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mul<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mul_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn div<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn div_scalar<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn matmul<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn neg<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D2> {
        todo!()
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D>,
        value: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn index_select<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        dim: usize,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn index_select_assign<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        dim: usize,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<1>,
        value: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D1> {
        todo!()
    }

    fn mask_scatter<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        source: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mask_fill<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        mask: <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D>,
        value: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn equal<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn equal_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal_elem<const D: usize>(
        lhs: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        rhs: <WGPUBackend<G, F, I> as Backend>::FloatElem,
    ) -> <WGPUBackend<G, F, I> as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn sum<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn sum_dim<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn mean<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn mean_dim<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        tensor: &<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <<WGPUBackend<G, F, I> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>
    {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        tensor: <<WGPUBackend<G, F, I> as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn exp<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn log<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn log1p<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn powf<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        value: f32,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sqrt<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cos<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sin<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn tanh<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn erf<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn cat<const D: usize>(
        tensors: Vec<<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn argmax<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }

    fn argmin<const D: usize>(
        tensor: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<D>,
        dim: usize,
    ) -> <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<D> {
        todo!()
    }
}
