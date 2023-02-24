use crate::tensor::{ADTensor, IntTensor};
use crate::ADBackendDecorator;

use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

impl<B: Backend> ModuleOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn embedding(weights: ADTensor<B, 2>, indexes: IntTensor<B, 2>) -> ADTensor<B, 3> {
        todo!();
    }

    fn embedding_backward(
        weights: ADTensor<B, 2>,
        output: ADTensor<B, 3>,
        indexes: IntTensor<B, 2>,
    ) -> ADTensor<B, 2> {
        todo!();
    }

    fn conv2d(
        x: ADTensor<B, 4>,
        weight: ADTensor<B, 4>,
        bias: Option<ADTensor<B, 1>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        todo!();
    }
    fn conv1d(
        x: ADTensor<B, 3>,
        weight: ADTensor<B, 3>,
        bias: Option<ADTensor<B, 1>>,
        stride: usize,
        padding: usize,
    ) -> ADTensor<B, 3> {
        todo!();
    }

    fn max_pool2d(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        todo!();
    }

    fn max_pool2d_with_indexes(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> MaxPool2dWithIndexes<ADBackendDecorator<B>> {
        todo!();
    }

    fn max_pool2d_with_indexes_backward(
        x: ADTensor<B, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        output_grad: ADTensor<B, 4>,
        indexes: IntTensor<B, 4>,
    ) -> MaxPool2dBackward<ADBackendDecorator<B>> {
        todo!();
    }
}
