use crate::{
    Fusion, FusionBackend,
    client::FusionClient,
    stream::{OperationStreams, execution::Operation},
};
use burn_ir::*;
use burn_tensor::{
    Element, Shape,
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, FloatTensor,
        IntTensor, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps,
        conv::{
            calculate_conv_output_size, calculate_conv_transpose_output_size,
            calculate_pool_output_size,
        },
    },
};
use std::marker::PhantomData;

macro_rules! make_ops {
    ($name:ident, $desc:ty, $fn:expr) => {
        #[derive(new, Debug)]
        struct $name<B: FusionBackend> {
            desc: $desc,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                #[allow(clippy::redundant_closure_call)]
                $fn(&self.desc, handles)
            }
        }
    };
}

impl<B: FusionBackend> ModuleOps<Fusion<B>> for Fusion<B> {
    fn conv1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self> {
        make_ops!(Conv1dOps, Conv1dOpIr, |desc: &Conv1dOpIr,
                                          handles: &mut HandleContainer<
            B::Handle,
        >| {
            let x = handles.get_float_tensor::<B>(&desc.x);
            let weight = handles.get_float_tensor::<B>(&desc.weight);
            let bias = desc
                .bias
                .as_ref()
                .map(|bias| handles.get_float_tensor::<B>(bias));
            let output = B::conv1d(x, weight, bias, desc.options.clone().into());
            handles.register_float_tensor::<B>(&desc.out.id, output);
        });

        let size = calculate_conv_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.dilation[0],
            x.shape[2],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let shape = vec![x.shape[0], weight.shape[0], size];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let description = Conv1dOpIr {
            x: x.into_ir(),
            weight: weight.into_ir(),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.clone().register(
            streams,
            OperationIr::Module(ModuleOperationIr::Conv1d(description.clone())),
            Conv1dOps::<B>::new(description),
        );

        out
    }

    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        make_ops!(Conv2dOps, Conv2dOpIr, |args: &Conv2dOpIr,
                                          handles: &mut HandleContainer<
            B::Handle,
        >| {
            let x = handles.get_float_tensor::<B>(&args.x);
            let weight = handles.get_float_tensor::<B>(&args.weight);
            let bias = args
                .bias
                .as_ref()
                .map(|bias| handles.get_float_tensor::<B>(bias));

            let output = B::conv2d(x, weight, bias, args.options.clone().into());

            handles.register_float_tensor::<B>(&args.out.id, output);
        });

        let size_0 = calculate_conv_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_conv_output_size(
            weight.shape[3],
            options.stride[1],
            options.padding[1],
            options.dilation[1],
            x.shape[3],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }
        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = Conv2dOpIr {
            x: x.into_ir(),
            weight: weight.into_ir(),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::Conv2d(desc.clone())),
            Conv2dOps::<B>::new(desc),
        );

        out
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        make_ops!(
            DeformConv2dOps,
            DeformConv2dOpIr,
            |args: &DeformConv2dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let offset = handles.get_float_tensor::<B>(&args.offset);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let mask = args
                    .mask
                    .as_ref()
                    .map(|mask| handles.get_float_tensor::<B>(mask));
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));

                let output =
                    B::deform_conv2d(x, offset, weight, mask, bias, args.options.clone().into());

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size_0 = calculate_conv_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_conv_output_size(
            weight.shape[3],
            options.stride[1],
            options.padding[1],
            options.dilation[1],
            x.shape[3],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&offset);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }
        if let Some(mask) = mask.as_ref() {
            streams.tensor(mask)
        }

        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = DeformConv2dOpIr {
            x: x.into_ir(),
            offset: offset.into_ir(),
            weight: weight.into_ir(),
            mask: mask.map(|mask| mask.into_ir()),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::DeformableConv2d(Box::new(desc.clone()))),
            DeformConv2dOps::<B>::new(desc),
        );

        out
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
        make_ops!(
            DeformConv2dBackwardOps,
            DeformConv2dBackwardOpIr,
            |args: &DeformConv2dBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let offset = handles.get_float_tensor::<B>(&args.offset);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let mask = args
                    .mask
                    .as_ref()
                    .map(|mask| handles.get_float_tensor::<B>(mask));
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));
                let output_grad = handles.get_float_tensor::<B>(&args.out_grad);

                let output = B::deform_conv2d_backward(
                    x,
                    offset,
                    weight,
                    mask,
                    bias,
                    output_grad,
                    args.options.clone().into(),
                );

                handles.register_float_tensor::<B>(&args.input_grad.id, output.x_grad);
                handles.register_float_tensor::<B>(&args.offset_grad.id, output.offset_grad);
                handles.register_float_tensor::<B>(&args.weight_grad.id, output.weight_grad);
                if let Some((mask_grad, field)) = output.mask_grad.zip(args.mask_grad.as_ref()) {
                    handles.register_float_tensor::<B>(&field.id, mask_grad);
                }
                if let Some((bias_grad, field)) = output.bias_grad.zip(args.bias_grad.as_ref()) {
                    handles.register_float_tensor::<B>(&field.id, bias_grad);
                }
            }
        );

        let input_grad = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());
        let offset_grad = offset
            .client
            .tensor_uninitialized(offset.shape.clone(), B::FloatElem::dtype());
        let weight_grad = offset
            .client
            .tensor_uninitialized(weight.shape.clone(), B::FloatElem::dtype());
        let mask_grad = mask.as_ref().map(|mask| {
            offset
                .client
                .tensor_uninitialized(mask.shape.clone(), B::FloatElem::dtype())
        });
        let bias_grad = bias.as_ref().map(|bias| {
            offset
                .client
                .tensor_uninitialized(bias.shape.clone(), B::FloatElem::dtype())
        });

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&offset);
        streams.tensor(&weight);
        streams.tensor(&output_grad);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }
        if let Some(mask) = mask.as_ref() {
            streams.tensor(mask)
        }

        let desc = DeformConv2dBackwardOpIr {
            x: x.into_ir(),
            offset: offset.into_ir(),
            weight: weight.into_ir(),
            mask: mask.map(|mask| mask.into_ir()),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out_grad: output_grad.into_ir(),
            input_grad: input_grad.to_ir_out(),
            offset_grad: offset_grad.to_ir_out(),
            weight_grad: weight_grad.to_ir_out(),
            mask_grad: mask_grad.as_ref().map(|mask_grad| mask_grad.to_ir_out()),
            bias_grad: bias_grad.as_ref().map(|bias_grad| bias_grad.to_ir_out()),
        };

        input_grad.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::DeformableConv2dBackward(Box::new(
                desc.clone(),
            ))),
            DeformConv2dBackwardOps::<B>::new(desc),
        );

        DeformConv2dBackward::new(input_grad, offset_grad, weight_grad, mask_grad, bias_grad)
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        make_ops!(Conv3dOps, Conv3dOpIr, |args: &Conv3dOpIr,
                                          handles: &mut HandleContainer<
            B::Handle,
        >| {
            let x = handles.get_float_tensor::<B>(&args.x);
            let weight = handles.get_float_tensor::<B>(&args.weight);
            let bias = args
                .bias
                .as_ref()
                .map(|bias| handles.get_float_tensor::<B>(bias));

            let output = B::conv3d(x, weight, bias, args.options.clone().into());

            handles.register_float_tensor::<B>(&args.out.id, output);
        });

        let size_0 = calculate_conv_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_conv_output_size(
            weight.shape[3],
            options.stride[1],
            options.padding[1],
            options.dilation[1],
            x.shape[3],
        );
        let size_2 = calculate_conv_output_size(
            weight.shape[4],
            options.stride[2],
            options.padding[2],
            options.dilation[2],
            x.shape[4],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1, size_2];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = Conv3dOpIr {
            x: x.into_ir(),
            weight: weight.into_ir(),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::Conv3d(desc.clone())),
            Conv3dOps::<B>::new(desc),
        );

        out
    }

    fn conv_transpose1d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self> {
        make_ops!(
            ConvTranspose1dOps,
            ConvTranspose1dOpIr,
            |args: &ConvTranspose1dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));

                let output = B::conv_transpose1d(x, weight, bias, args.options.clone().into());

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size = calculate_conv_transpose_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.padding_out[0],
            options.dilation[0],
            x.shape[2],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let shape = vec![x.shape[0], weight.shape[1] * options.groups, size];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = ConvTranspose1dOpIr {
            x: x.into_ir(),
            weight: weight.into_ir(),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::ConvTranspose1d(desc.clone())),
            ConvTranspose1dOps::<B>::new(desc),
        );

        out
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        make_ops!(
            ConvTranspose2dOps,
            ConvTranspose2dOpIr,
            |args: &ConvTranspose2dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));

                let output = B::conv_transpose2d(x, weight, bias, args.options.clone().into());

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size_0 = calculate_conv_transpose_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.padding_out[0],
            options.dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_conv_transpose_output_size(
            weight.shape[3],
            options.stride[1],
            options.padding[1],
            options.padding_out[1],
            options.dilation[1],
            x.shape[3],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let shape = vec![x.shape[0], weight.shape[1] * options.groups, size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = ConvTranspose2dOpIr {
            x: x.into_ir(),
            weight: weight.into_ir(),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::ConvTranspose2d(desc.clone())),
            ConvTranspose2dOps::<B>::new(desc),
        );

        out
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        make_ops!(
            ConvTranspose3dOps,
            ConvTranspose3dOpIr,
            |args: &ConvTranspose3dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));

                let output = B::conv_transpose3d(x, weight, bias, args.options.clone().into());

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size_0 = calculate_conv_transpose_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.padding_out[0],
            options.dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_conv_transpose_output_size(
            weight.shape[3],
            options.stride[1],
            options.padding[1],
            options.padding_out[1],
            options.dilation[1],
            x.shape[3],
        );
        let size_2 = calculate_conv_transpose_output_size(
            weight.shape[4],
            options.stride[2],
            options.padding[2],
            options.padding_out[2],
            options.dilation[2],
            x.shape[4],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&weight);

        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let shape = vec![
            x.shape[0],
            weight.shape[1] * options.groups,
            size_0,
            size_1,
            size_2,
        ];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = ConvTranspose3dOpIr {
            x: x.into_ir(),
            weight: weight.into_ir(),
            bias: bias.map(|bias| bias.into_ir()),
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::ConvTranspose3d(desc.clone())),
            ConvTranspose3dOps::<B>::new(desc),
        );

        out
    }

    fn avg_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        make_ops!(
            AvgPool1dOps,
            AvgPool1dOpIr,
            |args: &AvgPool1dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::avg_pool1d(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.count_include_pad,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let size = calculate_pool_output_size(kernel_size, stride, padding, 1, x.shape[2]);
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = AvgPool1dOpIr {
            x: x.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AvgPool1d(desc.clone())),
            AvgPool1dOps::<B>::new(desc),
        );

        out
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        make_ops!(
            AvgPool2dOps,
            AvgPool2dOpIr,
            |args: &AvgPool2dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::avg_pool2d(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.count_include_pad,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size_0 =
            calculate_pool_output_size(kernel_size[0], stride[0], padding[0], 1, x.shape[2]);
        let size_1 =
            calculate_pool_output_size(kernel_size[1], stride[1], padding[1], 1, x.shape[3]);

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = AvgPool2dOpIr {
            x: x.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AvgPool2d(desc.clone())),
            AvgPool2dOps::<B>::new(desc),
        );

        out
    }

    fn avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        make_ops!(
            AvgPool1dBackwardOps,
            AvgPool1dBackwardOpIr,
            |args: &AvgPool1dBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output = B::avg_pool1d_backward(
                    x,
                    grad,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.count_include_pad,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&grad);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = AvgPool1dBackwardOpIr {
            x: x.into_ir(),
            grad: grad.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AvgPool1dBackward(desc.clone())),
            AvgPool1dBackwardOps::<B>::new(desc),
        );

        out
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self> {
        make_ops!(
            AvgPool2dBackwardOps,
            AvgPool2dBackwardOpIr,
            |args: &AvgPool2dBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output = B::avg_pool2d_backward(
                    x,
                    grad,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.count_include_pad,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&grad);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = AvgPool2dBackwardOpIr {
            x: x.into_ir(),
            grad: grad.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AvgPool2dBackward(desc.clone())),
            AvgPool2dBackwardOps::<B>::new(desc),
        );

        out
    }

    fn max_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<Self> {
        make_ops!(
            MaxPool1dOps,
            MaxPool1dOpIr,
            |args: &MaxPool1dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::max_pool1d(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size = calculate_pool_output_size(kernel_size, stride, padding, dilation, x.shape[2]);

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = MaxPool1dOpIr {
            x: x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::MaxPool1d(desc.clone())),
            MaxPool1dOps::<B>::new(desc),
        );

        out
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self> {
        make_ops!(
            MaxPool2dOps,
            MaxPool2dOpIr,
            |args: &MaxPool2dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::max_pool2d(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let size_0 = calculate_pool_output_size(
            kernel_size[0],
            stride[0],
            padding[0],
            dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_pool_output_size(
            kernel_size[1],
            stride[1],
            padding[1],
            dilation[1],
            x.shape[3],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = MaxPool2dOpIr {
            x: x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::MaxPool2d(desc.clone())),
            MaxPool2dOps::<B>::new(desc),
        );

        out
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<Self> {
        make_ops!(
            MaxPool1dWithIndicesOps,
            MaxPool1dWithIndicesOpIr,
            |args: &MaxPool1dWithIndicesOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::max_pool1d_with_indices(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.output);
                handles.register_int_tensor::<B>(&args.out_indices.id, output.indices);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let size = calculate_pool_output_size(kernel_size, stride, padding, dilation, x.shape[2]);
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape.clone()), B::FloatElem::dtype());
        let out_indices = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::IntElem::dtype());

        let desc = MaxPool1dWithIndicesOpIr {
            x: x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::MaxPool1dWithIndices(desc.clone())),
            MaxPool1dWithIndicesOps::<B>::new(desc),
        );

        MaxPool1dWithIndices::new(out, out_indices)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        make_ops!(
            MaxPool2dWithIndicesOps,
            MaxPool2dWithIndicesOpIr,
            |args: &MaxPool2dWithIndicesOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::max_pool2d_with_indices(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.output);
                handles.register_int_tensor::<B>(&args.out_indices.id, output.indices);
            }
        );

        let size_0 = calculate_pool_output_size(
            kernel_size[0],
            stride[0],
            padding[0],
            dilation[0],
            x.shape[2],
        );
        let size_1 = calculate_pool_output_size(
            kernel_size[1],
            stride[1],
            padding[1],
            dilation[1],
            x.shape[3],
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape.clone()), B::FloatElem::dtype());
        let out_indices = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::IntElem::dtype());

        let desc = MaxPool2dWithIndicesOpIr {
            x: x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::MaxPool2dWithIndices(desc.clone())),
            MaxPool2dWithIndicesOps::<B>::new(desc),
        );

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
        make_ops!(
            MaxPool1dWithIndicesBackwardOps,
            MaxPool1dWithIndicesBackwardOpIr,
            |args: &MaxPool1dWithIndicesBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let indices = handles.get_int_tensor::<B>(&args.indices);
                let output = B::max_pool1d_with_indices_backward(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                    grad,
                    indices,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.x_grad);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&output_grad);
        streams.tensor(&indices);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = MaxPool1dWithIndicesBackwardOpIr {
            x: x.into_ir(),
            grad: output_grad.into_ir(),
            indices: indices.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::MaxPool1dWithIndicesBackward(
                desc.clone(),
            )),
            MaxPool1dWithIndicesBackwardOps::<B>::new(desc),
        );

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
        make_ops!(
            MaxPool2dWithIndicesBackwardOps,
            MaxPool2dWithIndicesBackwardOpIr,
            |args: &MaxPool2dWithIndicesBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let indices = handles.get_int_tensor::<B>(&args.indices);
                let output = B::max_pool2d_with_indices_backward(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                    grad,
                    indices,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.x_grad);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&output_grad);
        streams.tensor(&indices);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = MaxPool2dWithIndicesBackwardOpIr {
            x: x.into_ir(),
            grad: output_grad.into_ir(),
            indices: indices.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::MaxPool2dWithIndicesBackward(
                desc.clone(),
            )),
            MaxPool2dWithIndicesBackwardOps::<B>::new(desc),
        );

        MaxPool2dBackward::new(out)
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self>, output_size: usize) -> FloatTensor<Self> {
        make_ops!(
            AdaptiveAvgPool1dOps,
            AdaptiveAvgPool1dOpIr,
            |args: &AdaptiveAvgPool1dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::adaptive_avg_pool1d(x, args.output_size);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], output_size];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = AdaptiveAvgPool1dOpIr {
            x: x.into_ir(),
            output_size,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool1d(desc.clone())),
            AdaptiveAvgPool1dOps::<B>::new(desc),
        );

        out
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        make_ops!(
            AdaptiveAvgPool2dOps,
            AdaptiveAvgPool2dOpIr,
            |args: &AdaptiveAvgPool2dOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::adaptive_avg_pool2d(x, args.output_size);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], output_size[0], output_size[1]];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = AdaptiveAvgPool2dOpIr {
            x: x.into_ir(),
            output_size,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2d(desc.clone())),
            AdaptiveAvgPool2dOps::<B>::new(desc),
        );

        out
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        make_ops!(
            AdaptiveAvgPool1dBackwardOps,
            AdaptiveAvgPool1dBackwardOpIr,
            |args: &AdaptiveAvgPool1dBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output = B::adaptive_avg_pool1d_backward(x, grad);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&grad);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());
        let desc = AdaptiveAvgPool1dBackwardOpIr {
            x: x.into_ir(),
            grad: grad.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool1dBackward(desc.clone())),
            AdaptiveAvgPool1dBackwardOps::<B>::new(desc),
        );

        out
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        make_ops!(
            AdaptiveAvgPool2dBackwardOps,
            AdaptiveAvgPool2dBackwardOpIr,
            |args: &AdaptiveAvgPool2dBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output = B::adaptive_avg_pool2d_backward(x, grad);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&grad);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = AdaptiveAvgPool2dBackwardOpIr {
            x: x.into_ir(),
            grad: grad.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2dBackward(desc.clone())),
            AdaptiveAvgPool2dBackwardOps::<B>::new(desc),
        );

        out
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        make_ops!(
            InterpolateOps,
            InterpolateOpIr,
            |args: &InterpolateOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::interpolate(x, args.output_size, args.options.clone().into());
                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);

        let shape = vec![x.shape[0], x.shape[1], output_size[0], output_size[1]];
        let out = x
            .client
            .tensor_uninitialized(Shape::from(shape), B::FloatElem::dtype());

        let desc = InterpolateOpIr {
            x: x.into_ir(),
            output_size,
            options: options.into(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::Interpolate(desc.clone())),
            InterpolateOps::<B>::new(desc),
        );

        out
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        make_ops!(
            InterpolateBackwardOps,
            InterpolateBackwardOpIr,
            |args: &InterpolateBackwardOpIr, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output =
                    B::interpolate_backward(x, grad, args.output_size, args.options.clone().into());

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let mut streams = OperationStreams::default();
        streams.tensor(&x);
        streams.tensor(&grad);

        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = InterpolateBackwardOpIr {
            x: x.into_ir(),
            grad: grad.into_ir(),
            output_size,
            options: options.into(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Module(ModuleOperationIr::InterpolateBackward(desc.clone())),
            InterpolateBackwardOps::<B>::new(desc),
        );
        out
    }
}
