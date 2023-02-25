use crate::tensor::{ADTensor, IntTensor};
use crate::ADBackendDecorator;

use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

impl<B: Backend> ModuleOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn embedding(_weights: ADTensor<B, 2>, _indexes: IntTensor<B, 2>) -> ADTensor<B, 3> {
        todo!();
    }

    fn embedding_backward(
        _weights: ADTensor<B, 2>,
        _output: ADTensor<B, 3>,
        _indexes: IntTensor<B, 2>,
    ) -> ADTensor<B, 2> {
        todo!();
    }

    fn conv2d(
        _x: ADTensor<B, 4>,
        _weight: ADTensor<B, 4>,
        _bias: Option<ADTensor<B, 1>>,
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        todo!();
    }
    fn conv1d(
        _x: ADTensor<B, 3>,
        _weight: ADTensor<B, 3>,
        _bias: Option<ADTensor<B, 1>>,
        _stride: usize,
        _padding: usize,
    ) -> ADTensor<B, 3> {
        todo!();
    }

    fn max_pool2d(
        _x: ADTensor<B, 4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> ADTensor<B, 4> {
        todo!();
    }

    fn max_pool2d_with_indexes(
        _x: ADTensor<B, 4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
    ) -> MaxPool2dWithIndexes<ADBackendDecorator<B>> {
        todo!();
    }

    fn max_pool2d_with_indexes_backward(
        _x: ADTensor<B, 4>,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _output_grad: ADTensor<B, 4>,
        _indexes: IntTensor<B, 4>,
    ) -> MaxPool2dBackward<ADBackendDecorator<B>> {
        todo!();
    }
}
