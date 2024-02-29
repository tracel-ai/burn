use burn_tensor::{
    ops::{
        ConvOptions, ConvTransposeOptions, FloatTensor, IntTensor, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps, UnfoldOptions,
    },
    Shape,
};
use candle_core::ToUsize2;

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    ops::base::reshape,
    Candle, CandleTensor,
};

impl<F: FloatCandleElement, I: IntCandleElement> ModuleOps<Self> for Candle<F, I> {
    fn conv1d(
        x: FloatTensor<Self, 3>,
        weight: FloatTensor<Self, 3>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self, 3> {
        let conv = x
            .tensor
            .conv1d(
                &weight.tensor,
                options.padding[0],
                options.stride[0],
                options.dilation[0],
                options.groups,
            )
            .unwrap();
        CandleTensor::new(match bias {
            Some(bias) => conv
                .broadcast_add(&bias.tensor.unsqueeze(1).unwrap())
                .unwrap(),
            None => conv,
        })
    }

    fn conv2d(
        x: FloatTensor<Self, 4>,
        weight: FloatTensor<Self, 4>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self, 4> {
        assert!(
            options.dilation[0] == options.dilation[1]
                && options.padding[0] == options.padding[1]
                && options.stride[0] == options.stride[1],
            "Candle does not support per dimension options in convolutions"
        );
        let conv = x
            .tensor
            .conv2d(
                &weight.tensor,
                options.padding[0],
                options.stride[0],
                options.dilation[0],
                options.groups,
            )
            .unwrap();
        CandleTensor::new(match bias {
            Some(bias) => conv
                .broadcast_add(
                    &bias
                        .tensor
                        .unsqueeze(0)
                        .unwrap()
                        .unsqueeze(2)
                        .unwrap()
                        .unsqueeze(3)
                        .unwrap(),
                )
                .unwrap(),
            None => conv,
        })
    }

    fn conv_transpose1d(
        x: FloatTensor<Self, 3>,
        weight: FloatTensor<Self, 3>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self, 3> {
        let conv_transpose = x
            .tensor
            .conv_transpose1d(
                &weight.tensor,
                options.padding[0],
                options.padding_out[0],
                options.stride[0],
                options.dilation[0],
                options.groups,
            )
            .unwrap();
        CandleTensor::new(match bias {
            Some(bias) => conv_transpose
                .broadcast_add(&bias.tensor.unsqueeze(0).unwrap().unsqueeze(2).unwrap())
                .unwrap(),
            None => conv_transpose,
        })
    }

    fn conv_transpose2d(
        x: FloatTensor<Self, 4>,
        weight: FloatTensor<Self, 4>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self, 4> {
        assert!(
            options.dilation[0] == options.dilation[1]
                && options.padding[0] == options.padding[1]
                && options.padding_out[0] == options.padding_out[1]
                && options.stride[0] == options.stride[1],
            "Candle does not support per dimension options in transposed convolutions"
        );
        assert!(
            options.groups == 1,
            "Candle does not support groups in transposed convolutions"
        );
        let conv_transpose = x
            .tensor
            .conv_transpose2d(
                &weight.tensor,
                options.padding[0],
                options.padding_out[0],
                options.stride[0],
                options.dilation[0],
            )
            .unwrap();
        CandleTensor::new(match bias {
            Some(bias) => conv_transpose
                .broadcast_add(
                    &bias
                        .tensor
                        .unsqueeze(0)
                        .unwrap()
                        .unsqueeze(2)
                        .unwrap()
                        .unsqueeze(3)
                        .unwrap(),
                )
                .unwrap(),
            None => conv_transpose,
        })
    }

    fn avg_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self, 4> {
        assert!(
            padding[0] == 0 && padding[1] == 0,
            "Candle does not support padding in pooling"
        );
        assert!(
            count_include_pad,
            "Candle does not support excluding pad count in pooling"
        );
        CandleTensor::new(
            x.tensor
                .avg_pool2d_with_stride((kernel_size[0], kernel_size[1]), (stride[0], stride[1]))
                .unwrap(),
        )
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self, 4> {
        panic!("avg_pool2d_backward is not supported by Candle")
    }

    fn max_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        assert!(
            padding[0] == 0 && padding[1] == 0,
            "Candle does not support padding in pooling"
        );
        assert!(
            dilation[0] == 1 && dilation[1] == 1,
            "Candle does not support dilation in pooling"
        );
        CandleTensor::new(
            x.tensor
                .max_pool2d_with_stride((kernel_size[0], kernel_size[1]), (stride[0], stride[1]))
                .unwrap(),
        )
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Candle<F, I>> {
        panic!("max_pool2d_with_indices is not supported by Candle")
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self, 4>,
        indices: IntTensor<Self, 4>,
    ) -> MaxPool2dBackward<Candle<F, I>> {
        panic!("max_pool2d_with_indices_backward is not supported by Candle")
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<Self, 4>,
        output_size: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        panic!("adaptive_avg_pool2 is not supported by Candle")
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
    ) -> FloatTensor<Self, 4> {
        panic!("adaptive_avg_pool2d_backward is not supported by Candle")
    }
}
