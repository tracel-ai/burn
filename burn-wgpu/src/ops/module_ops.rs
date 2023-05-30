use burn_tensor::{backend::Backend, ops::ModuleOps};

use crate::{
    element::{FloatElement, IntElement},
    GraphicsAPI, WGPUBackend,
};

impl<G, F, I> ModuleOps<WGPUBackend<G, F, I>> for WGPUBackend<G, F, I>
where
    G: GraphicsAPI + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn embedding(
        _weights: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<2>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<3> {
        todo!()
    }

    fn embedding_backward(
        _weights: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<2>,
        _output: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<3>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<2> {
        todo!()
    }

    fn conv2d(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _weight: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _bias: Option<<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1>>,
        _options: burn_tensor::ops::ConvOptions<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn conv_transpose2d(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _weight: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _bias: Option<<WGPUBackend<G, F, I> as Backend>::TensorPrimitive<1>>,
        _options: burn_tensor::ops::ConvTransposeOptions<2>,
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn avg_pool2d(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn avg_pool2d_backward(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _grad: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn max_pool2d(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn max_pool2d_with_indexes(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> burn_tensor::ops::MaxPool2dWithIndexes<WGPUBackend<G, F, I>> {
        todo!()
    }

    fn max_pool2d_with_indexes_backward(
        _x: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _output_grad: <WGPUBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _indexes: <WGPUBackend<G, F, I> as Backend>::IntTensorPrimitive<4>,
    ) -> burn_tensor::ops::MaxPool2dBackward<WGPUBackend<G, F, I>> {
        todo!()
    }
}
