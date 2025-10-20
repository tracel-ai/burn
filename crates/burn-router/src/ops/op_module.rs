use alloc::boxed::Box;

use burn_ir::{
    AdaptiveAvgPool1dBackwardOpIr, AdaptiveAvgPool1dOpIr, AdaptiveAvgPool2dBackwardOpIr,
    AdaptiveAvgPool2dOpIr, AvgPool1dBackwardOpIr, AvgPool1dOpIr, AvgPool2dBackwardOpIr,
    AvgPool2dOpIr, Conv1dOpIr, Conv2dOpIr, Conv3dOpIr, ConvTranspose1dOpIr, ConvTranspose2dOpIr,
    ConvTranspose3dOpIr, DeformConv2dBackwardOpIr, DeformConv2dOpIr, InterpolateBackwardOpIr,
    InterpolateOpIr, MaxPool1dOpIr, MaxPool1dWithIndicesBackwardOpIr, MaxPool1dWithIndicesOpIr,
    MaxPool2dOpIr, MaxPool2dWithIndicesBackwardOpIr, MaxPool2dWithIndicesOpIr, ModuleOperationIr,
    OperationIr,
};
use burn_tensor::Element;
use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, FloatTensor,
    IntElem, IntTensor, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices,
    MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};

use crate::{BackendRouter, OperationOutput, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> ModuleOps<Self> for BackendRouter<R> {
    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = Conv1dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::Conv1d(desc)))
            .output()
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = Conv2dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::Conv2d(desc)))
            .output()
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = Conv3dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::Conv3d(desc)))
            .output()
    }

    fn conv_transpose1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = ConvTranspose1dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::ConvTranspose1d(
                desc,
            )))
            .output()
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = ConvTranspose2dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::ConvTranspose2d(
                desc,
            )))
            .output()
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = ConvTranspose3dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::ConvTranspose3d(
                desc,
            )))
            .output()
    }

    fn avg_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = AvgPool1dOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::AvgPool1d(desc)))
            .output()
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = AvgPool2dOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::AvgPool2d(desc)))
            .output()
    }

    fn avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = AvgPool1dBackwardOpIr::create(
            x.into_ir(),
            grad.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::AvgPool1dBackward(
                desc,
            )))
            .output()
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = AvgPool2dBackwardOpIr::create(
            x.into_ir(),
            grad.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::AvgPool2dBackward(
                desc,
            )))
            .output()
    }

    fn max_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc =
            MaxPool1dOpIr::create(x.into_ir(), kernel_size, stride, padding, dilation, || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::Module(ModuleOperationIr::MaxPool1d(desc)))
            .output()
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc =
            MaxPool2dOpIr::create(x.into_ir(), kernel_size, stride, padding, dilation, || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::Module(ModuleOperationIr::MaxPool2d(desc)))
            .output()
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<Self> {
        let client = x.client.clone();
        let desc = MaxPool1dWithIndicesOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            IntElem::<Self>::dtype(),
            || client.create_empty_handle(),
        );

        let (out, out_indices) = client
            .register(OperationIr::Module(
                ModuleOperationIr::MaxPool1dWithIndices(desc),
            ))
            .outputs();

        MaxPool1dWithIndices::new(out, out_indices)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        let client = x.client.clone();
        let desc = MaxPool2dWithIndicesOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            IntElem::<Self>::dtype(),
            || client.create_empty_handle(),
        );

        let (out, out_indices) = client
            .register(OperationIr::Module(
                ModuleOperationIr::MaxPool2dWithIndices(desc),
            ))
            .outputs();

        MaxPool2dWithIndices::new(out, out_indices)
    }

    fn max_pool1d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool1dBackward<Self> {
        let client = x.client.clone();

        let desc = MaxPool1dWithIndicesBackwardOpIr::create(
            x.into_ir(),
            output_grad.into_ir(),
            indices.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            || client.create_empty_handle(),
        );

        let out = client
            .register(OperationIr::Module(
                ModuleOperationIr::MaxPool1dWithIndicesBackward(desc),
            ))
            .output();

        MaxPool1dBackward::new(out)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let client = x.client.clone();

        let desc = MaxPool2dWithIndicesBackwardOpIr::create(
            x.into_ir(),
            output_grad.into_ir(),
            indices.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            || client.create_empty_handle(),
        );

        let out = client
            .register(OperationIr::Module(
                ModuleOperationIr::MaxPool2dWithIndicesBackward(desc),
            ))
            .output();

        MaxPool2dBackward::new(out)
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self>, output_size: usize) -> FloatTensor<Self> {
        let client = x.client.clone();

        let desc = AdaptiveAvgPool1dOpIr::create(x.into_ir(), output_size, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool1d(
                desc,
            )))
            .output()
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let client = x.client.clone();

        let desc = AdaptiveAvgPool2dOpIr::create(x.into_ir(), output_size, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2d(
                desc,
            )))
            .output()
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();

        let desc = AdaptiveAvgPool1dBackwardOpIr::create(x.into_ir(), grad.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Module(
                ModuleOperationIr::AdaptiveAvgPool1dBackward(desc),
            ))
            .output()
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();

        let desc = AdaptiveAvgPool2dBackwardOpIr::create(x.into_ir(), grad.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Module(
                ModuleOperationIr::AdaptiveAvgPool2dBackward(desc),
            ))
            .output()
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = InterpolateOpIr::create(x.into_ir(), output_size, options.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Module(ModuleOperationIr::Interpolate(desc)))
            .output()
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = InterpolateBackwardOpIr::create(
            x.into_ir(),
            grad.into_ir(),
            output_size,
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::InterpolateBackward(
                desc,
            )))
            .output()
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = DeformConv2dOpIr::create(
            x.into_ir(),
            offset.into_ir(),
            weight.into_ir(),
            mask.map(|mask| mask.into_ir()),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::Module(ModuleOperationIr::DeformableConv2d(
                Box::new(desc),
            )))
            .output()
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
        let client = x.client.clone();
        let has_bias = bias.is_some();
        let has_mask = mask.is_some();

        let desc = DeformConv2dBackwardOpIr::create(
            x.into_ir(),
            offset.into_ir(),
            weight.into_ir(),
            mask.map(|mask| mask.into_ir()),
            bias.map(|bias| bias.into_ir()),
            output_grad.into_ir(),
            options.into(),
            || client.create_empty_handle(),
        );
        let mut outputs = client
            .register(OperationIr::Module(
                ModuleOperationIr::DeformableConv2dBackward(Box::new(desc)),
            ))
            .into_iter();

        // When the number of outputs is variable, the order is important
        let input_grad = outputs.next().unwrap();
        let offset_grad = outputs.next().unwrap();
        let weight_grad = outputs.next().unwrap();
        let mask_grad = has_mask.then(|| outputs.next().unwrap());
        let bias_grad = has_bias.then(|| outputs.next().unwrap());

        DeformConv2dBackward::new(input_grad, offset_grad, weight_grad, mask_grad, bias_grad)
    }
}
