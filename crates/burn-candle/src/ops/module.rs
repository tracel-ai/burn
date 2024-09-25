use burn_tensor::{
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, FloatTensor,
        IntTensor, InterpolateMode, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices,
        ModuleOps, UnfoldOptions,
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
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
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
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
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

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        unimplemented!("Candle does not support deformable convolutions")
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        unimplemented!("Candle does not support deformable convolutions")
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        panic!("Candle does not support 3D convolutions");
    }

    fn conv_transpose1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
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
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
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

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        panic!("Candle does not support 3D transposed convolutions");
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
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
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        panic!("avg_pool2d_backward is not supported by Candle")
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self> {
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
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Candle<F, I>> {
        panic!("max_pool2d_with_indices is not supported by Candle")
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Candle<F, I>> {
        panic!("max_pool2d_with_indices_backward is not supported by Candle")
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        panic!("adaptive_avg_pool2 is not supported by Candle")
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        panic!("adaptive_avg_pool2d_backward is not supported by Candle")
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let tensor = match options.mode {
            InterpolateMode::Nearest => x
                .tensor
                .upsample_nearest2d(output_size[0], output_size[1])
                .unwrap(),
            InterpolateMode::Bilinear => {
                panic!("bilinear interpolation is not supported by Candle")
            }
            InterpolateMode::Bicubic => {
                panic!("bicubic interpolation is not supported by Candle")
            }
        };

        CandleTensor::new(tensor)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        panic!("interpolate_backward is not supported by Candle")
    }
}
