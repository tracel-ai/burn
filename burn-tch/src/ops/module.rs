use crate::{element::TchElement, to_tensor, TchBackend, TchTensor};
use burn_tensor::ops::{MaxPool2dBackward, MaxPool2dWithIndexes, ModuleOps};

impl<E: TchElement> ModuleOps<TchBackend<E>> for TchBackend<E> {
    fn embedding(weights: TchTensor<E, 2>, indexes: TchTensor<i64, 2>) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::embedding(&weights.tensor, &indexes.tensor, -1, false, false);

        to_tensor(tensor)
    }

    fn embedding_backward(
        weights: TchTensor<E, 2>,
        output: TchTensor<E, 3>,
        indexes: TchTensor<i64, 2>,
    ) -> TchTensor<E, 2> {
        let [n_embedding, _d_model] = weights.shape().dims;
        let tensor = tch::Tensor::embedding_backward(
            &output.tensor,
            &indexes.tensor,
            n_embedding as i64,
            -1,
            false,
            false,
        );

        to_tensor(tensor)
    }

    fn conv1d(
        x: TchTensor<E, 3>,
        weight: TchTensor<E, 3>,
        bias: Option<TchTensor<E, 1>>,
        stride: usize,
        padding: usize,
    ) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::conv1d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            &[stride as i64],
            &[padding as i64],
            &[1],
            1,
        );

        to_tensor(tensor)
    }

    fn conv2d(
        x: TchTensor<E, 4>,
        weight: TchTensor<E, 4>,
        bias: Option<TchTensor<E, 1>>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::conv2d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            &[stride[0] as i64, stride[1] as i64],
            &[padding[0] as i64, padding[1] as i64],
            &[1, 1],
            1,
        );

        to_tensor(tensor)
    }

    fn max_pool2d(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::max_pool2d(
            &x.tensor,
            &[kernel_size[0] as i64, kernel_size[1] as i64],
            &[stride[0] as i64, stride[1] as i64],
            &[padding[0] as i64, padding[1] as i64],
            &[1, 1],
            false,
        );

        to_tensor(tensor)
    }

    fn max_pool2d_with_indexes(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> MaxPool2dWithIndexes<TchBackend<E>> {
        let (tensor, indexes) = tch::Tensor::max_pool2d_with_indices(
            &x.tensor,
            &[kernel_size[0] as i64, kernel_size[1] as i64],
            &[stride[0] as i64, stride[1] as i64],
            &[padding[0] as i64, padding[1] as i64],
            &[1, 1],
            false,
        );

        MaxPool2dWithIndexes::new(to_tensor(tensor), to_tensor(indexes))
    }

    fn max_pool2d_with_indexes_backward(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        output_grad: TchTensor<E, 4>,
        indexes: TchTensor<i64, 4>,
    ) -> MaxPool2dBackward<TchBackend<E>> {
        let grad = tch::Tensor::max_pool2d_with_indices_backward(
            &x.tensor,
            &output_grad.tensor,
            &[kernel_size[0] as i64, kernel_size[1] as i64],
            &[stride[0] as i64, stride[1] as i64],
            &[padding[0] as i64, padding[1] as i64],
            &[1, 1],
            false,
            &indexes.tensor,
        );

        MaxPool2dBackward::new(to_tensor(grad))
    }
}
