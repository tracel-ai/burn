use burn_tensor::{
    backend::Backend,
    ops::{BoolTensorOps, ModuleOps, TensorOps},
};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend, WGPUDevice,
};

impl<G, F, I> ModuleOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn embedding(
        weights: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<2>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<3> {
        todo!()
    }

    fn embedding_backward(
        weights: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<2>,
        output: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<3>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<2> {
        todo!()
    }

    fn conv2d(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        weight: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        bias: Option<<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1>>,
        options: burn_tensor::ops::ConvOptions<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn conv_transpose2d(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        weight: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        bias: Option<<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1>>,
        options: burn_tensor::ops::ConvTransposeOptions<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn avg_pool2d(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn avg_pool2d_backward(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        grad: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn max_pool2d(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn max_pool2d_with_indexes(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> burn_tensor::ops::MaxPool2dWithIndexes<WGPUBackend<G, F, I>> {
        todo!()
    }

    fn max_pool2d_with_indexes_backward(
        x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        output_grad: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<4>,
    ) -> burn_tensor::ops::MaxPool2dBackward<WGPUBackend<G, F, I>> {
        todo!()
    }
}
