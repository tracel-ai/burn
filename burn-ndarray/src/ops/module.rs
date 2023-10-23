use super::{
    adaptive_avgpool::{adaptive_avg_pool2d, adaptive_avg_pool2d_backward},
    avgpool::{avg_pool2d, avg_pool2d_backward},
    conv::{conv2d, conv_transpose2d},
    maxpool::{max_pool2d, max_pool2d_backward, max_pool2d_with_indices},
};
use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend};
use burn_tensor::ops::*;

impl<E: FloatNdArrayElement> ModuleOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn conv2d(
        x: NdArrayTensor<E, 4>,
        weight: NdArrayTensor<E, 4>,
        bias: Option<NdArrayTensor<E, 1>>,
        options: ConvOptions<2>,
    ) -> NdArrayTensor<E, 4> {
        conv2d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: NdArrayTensor<E, 4>,
        weight: NdArrayTensor<E, 4>,
        bias: Option<NdArrayTensor<E, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> NdArrayTensor<E, 4> {
        conv_transpose2d(x, weight, bias, options)
    }

    fn avg_pool2d(
        x: NdArrayTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> NdArrayTensor<E, 4> {
        avg_pool2d(x, kernel_size, stride, padding, count_include_pad)
    }

    fn avg_pool2d_backward(
        x: NdArrayTensor<E, 4>,
        grad: NdArrayTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> NdArrayTensor<E, 4> {
        avg_pool2d_backward(x, grad, kernel_size, stride, padding, count_include_pad)
    }

    fn max_pool2d(
        x: NdArrayTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> NdArrayTensor<E, 4> {
        max_pool2d(x, kernel_size, stride, padding, dilation)
    }

    fn max_pool2d_with_indices(
        x: NdArrayTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<NdArrayBackend<E>> {
        let (output, indices) = max_pool2d_with_indices(x, kernel_size, stride, padding, dilation);

        MaxPool2dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: NdArrayTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: NdArrayTensor<E, 4>,
        indices: NdArrayTensor<i64, 4>,
    ) -> MaxPool2dBackward<NdArrayBackend<E>> {
        MaxPool2dBackward::new(max_pool2d_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            output_grad,
            indices,
        ))
    }

    fn adaptive_avg_pool2d(x: NdArrayTensor<E, 4>, output_size: [usize; 2]) -> NdArrayTensor<E, 4> {
        adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: NdArrayTensor<E, 4>,
        grad: NdArrayTensor<E, 4>,
    ) -> NdArrayTensor<E, 4> {
        adaptive_avg_pool2d_backward(x, grad)
    }
}
