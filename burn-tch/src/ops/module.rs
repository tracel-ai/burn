use crate::{backend, element::TchElement, TchBackend, TchTensor};
use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, MaxPool1dWithIndices, MaxPool2dBackward,
    MaxPool2dWithIndices, ModuleOps, UnfoldOptions,
};
use tch::{Device, IndexOp, Kind};

impl<E: TchElement> ModuleOps<TchBackend<E>> for TchBackend<E> {
    fn embedding(weights: TchTensor<E, 2>, indices: TchTensor<i64, 2>) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::embedding(&weights.tensor, &indices.tensor, -1, false, false);

        TchTensor::new(tensor)
    }

    fn embedding_backward(
        weights: TchTensor<E, 2>,
        output: TchTensor<E, 3>,
        indices: TchTensor<i64, 2>,
    ) -> TchTensor<E, 2> {
        let [n_embedding, _d_model] = weights.shape().dims;
        let tensor = tch::Tensor::embedding_backward(
            &output.tensor,
            &indices.tensor,
            n_embedding as i64,
            -1,
            false,
            false,
        );

        TchTensor::new(tensor)
    }

    fn conv1d(
        x: TchTensor<E, 3>,
        weight: TchTensor<E, 3>,
        bias: Option<TchTensor<E, 1>>,
        options: ConvOptions<1>,
    ) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::conv1d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.dilation.map(|i| i as i64),
            options.groups as i64,
        );

        TchTensor::new(tensor)
    }

    fn conv2d(
        x: TchTensor<E, 4>,
        weight: TchTensor<E, 4>,
        bias: Option<TchTensor<E, 1>>,
        options: ConvOptions<2>,
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::conv2d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.dilation.map(|i| i as i64),
            options.groups as i64,
        );

        TchTensor::new(tensor)
    }

    fn conv_transpose2d(
        x: TchTensor<E, 4>,
        weight: TchTensor<E, 4>,
        bias: Option<TchTensor<E, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::conv_transpose2d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.padding_out.map(|i| i as i64),
            options.groups as i64,
            options.dilation.map(|i| i as i64),
        );

        TchTensor::new(tensor)
    }

    fn conv_transpose1d(
        x: TchTensor<E, 3>,
        weight: TchTensor<E, 3>,
        bias: Option<TchTensor<E, 1>>,
        options: ConvTransposeOptions<1>,
    ) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::conv_transpose1d(
            &x.tensor,
            &weight.tensor,
            bias.map(|t| t.tensor),
            options.stride.map(|i| i as i64),
            options.padding.map(|i| i as i64),
            options.padding_out.map(|i| i as i64),
            options.groups as i64,
            options.dilation.map(|i| i as i64),
        );

        TchTensor::new(tensor)
    }

    fn unfold4d(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        options: UnfoldOptions,
    ) -> TchTensor<E, 3> {
        // Need to nest this function for creating the specialized weight
        // matrix to have conv2d perform the sliding window mechanism for us.
        fn create_unfolding_weight(in_channels: i64, kernel_size: [i64; 2]) -> tch::Tensor {
            let weight = tch::Tensor::zeros(
                [
                    in_channels * kernel_size[0] * kernel_size[1],
                    in_channels,
                    kernel_size[0],
                    kernel_size[1],
                ],
                (Kind::Float, Device::Cpu),
            );

            for k in 0..in_channels {
                for i in 0..kernel_size[0] {
                    for j in 0..kernel_size[1] {
                        let output_channel =
                            k * kernel_size[0] * kernel_size[1] + i * kernel_size[1] + j;
                        let _ = weight.i((output_channel, k, i, j)).fill_(1.0);
                    }
                }
            }

            weight
        }

        let batch_size = x.shape().dims[0];
        let channels_in = x.shape().dims[1];
        let stride = options.stride.unwrap_or([1, 1]);
        let padding = options.padding.unwrap_or([0, 0]);
        let dilation = options.dilation.unwrap_or([1, 1]);

        let channels_out = channels_in * kernel_size[0] * kernel_size[1];
        // See https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold for full explanation,
        // This calculates the number of patches with each patch having channels_out values
        let l_dim_1 = (x.shape().dims[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            / stride[0]
            + 1;
        let l_dim_2 = (x.shape().dims[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            / stride[1]
            + 1;
        let l = l_dim_1 * l_dim_2;

        let weight = TchTensor::new(create_unfolding_weight(
            channels_in as i64,
            [kernel_size[0] as i64, kernel_size[1] as i64],
        ));
        let unfolded: TchTensor<E, 4> =
            <backend::TchBackend<E> as ModuleOps<TchBackend<E>>>::conv2d(
                x,
                weight,
                None,
                ConvOptions {
                    stride,
                    padding,
                    dilation,
                    groups: 1,
                },
            );

        let reshaped = tch::Tensor::zeros(
            [batch_size as i64, channels_out as i64, l as i64],
            (Kind::Float, Device::Cpu),
        );

        // Iterate over each dimension and fill in the values from unfolded to reshaped
        for b in 0..batch_size {
            for c in 0..channels_out {
                let mut l_index: usize = 0;
                for h in 0..l_dim_1 {
                    for w in 0..l_dim_2 {
                        let value = unfolded.tensor.i((b as i64, c as i64, h as i64, w as i64));
                        reshaped
                            .i((b as i64, c as i64, l_index as i64))
                            .copy_(&value);

                        l_index += 1;
                    }
                }
            }
        }

        TchTensor::new(reshaped)
    }

    fn avg_pool1d(
        x: TchTensor<E, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::avg_pool1d(
            &x.tensor,
            [kernel_size as i64],
            [stride as i64],
            [padding as i64],
            false,
            count_include_pad,
        );

        TchTensor::new(tensor)
    }
    fn avg_pool2d(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::avg_pool2d(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            false,
            count_include_pad,
            None,
        );

        TchTensor::new(tensor)
    }

    fn avg_pool2d_backward(
        x: TchTensor<E, 4>,
        grad: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::avg_pool2d_backward(
            &x.tensor,
            &grad.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            false,
            count_include_pad,
            None,
        );

        TchTensor::new(tensor)
    }

    fn max_pool1d(
        x: TchTensor<E, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::max_pool1d(
            &x.tensor,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            false,
        );

        TchTensor::new(tensor)
    }

    fn max_pool1d_with_indices(
        x: TchTensor<E, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<TchBackend<E>> {
        let (tensor, indices) = tch::Tensor::max_pool1d_with_indices(
            &x.tensor,
            kernel_size as i64,
            stride as i64,
            padding as i64,
            dilation as i64,
            false,
        );

        MaxPool1dWithIndices::new(TchTensor::new(tensor), TchTensor::new(indices))
    }

    fn max_pool2d(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::max_pool2d(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            false,
        );

        TchTensor::new(tensor)
    }

    fn max_pool2d_with_indices(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<TchBackend<E>> {
        let (tensor, indices) = tch::Tensor::max_pool2d_with_indices(
            &x.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            false,
        );

        MaxPool2dWithIndices::new(TchTensor::new(tensor), TchTensor::new(indices))
    }

    fn max_pool2d_with_indices_backward(
        x: TchTensor<E, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: TchTensor<E, 4>,
        indices: TchTensor<i64, 4>,
    ) -> MaxPool2dBackward<TchBackend<E>> {
        let grad = tch::Tensor::max_pool2d_with_indices_backward(
            &x.tensor,
            &output_grad.tensor,
            [kernel_size[0] as i64, kernel_size[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding[0] as i64, padding[1] as i64],
            [dilation[0] as i64, dilation[1] as i64],
            false,
            &indices.tensor,
        );

        MaxPool2dBackward::new(TchTensor::new(grad))
    }

    fn adaptive_avg_pool2d(x: TchTensor<E, 4>, output_size: [usize; 2]) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::adaptive_avg_pool2d(&x.tensor, output_size.map(|e| e as i64));

        TchTensor::new(tensor)
    }

    fn adaptive_avg_pool2d_backward(x: TchTensor<E, 4>, grad: TchTensor<E, 4>) -> TchTensor<E, 4> {
        let tensor = tch::Tensor::internal_adaptive_avg_pool2d_backward(&x.tensor, &grad.tensor);

        TchTensor::new(tensor)
    }

    fn adaptive_avg_pool1d(x: TchTensor<E, 3>, output_size: usize) -> TchTensor<E, 3> {
        let tensor = tch::Tensor::adaptive_avg_pool1d(&x.tensor, output_size as i64);

        TchTensor::new(tensor)
    }
}
