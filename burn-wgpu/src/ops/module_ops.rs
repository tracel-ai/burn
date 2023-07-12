use burn_tensor::{backend::Backend, ops::ModuleOps};

use crate::{
    element::{FloatElement, IntElement},
    kernel, GraphicsApi, WgpuBackend,
};

use super::FloatTensor;

impl<G, F, I> ModuleOps<WgpuBackend<G, F, I>> for WgpuBackend<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn conv2d(
        _x: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _weight: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _bias: Option<<WgpuBackend<G, F, I> as Backend>::TensorPrimitive<1>>,
        _options: burn_tensor::ops::ConvOptions<2>,
    ) -> <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn conv_transpose2d(
        _x: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _weight: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _bias: Option<<WgpuBackend<G, F, I> as Backend>::TensorPrimitive<1>>,
        _options: burn_tensor::ops::ConvTransposeOptions<2>,
    ) -> <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn avg_pool2d(
        _x: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn avg_pool2d_backward(
        _x: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _grad: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4> {
        todo!()
    }

    fn max_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        kernel::pool::max_pool2d(x, kernel_size, stride, padding)
    }

    fn max_pool2d_with_indices(
        _x: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> burn_tensor::ops::MaxPool2dWithIndices<WgpuBackend<G, F, I>> {
        todo!()
    }

    fn max_pool2d_with_indices_backward(
        _x: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _output_grad: <WgpuBackend<G, F, I> as Backend>::TensorPrimitive<4>,
        _indices: <WgpuBackend<G, F, I> as Backend>::IntTensorPrimitive<4>,
    ) -> burn_tensor::ops::MaxPool2dBackward<WgpuBackend<G, F, I>> {
        todo!()
    }
}
