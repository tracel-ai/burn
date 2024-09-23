use crate::{client::FusionClient, stream::execution::Operation, Fusion, FusionBackend};
use burn_tensor::{
    ops::{
        conv::{
            calculate_conv_output_size, calculate_conv_transpose_output_size,
            calculate_pool_output_size,
        },
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, FloatTensor,
        IntTensor, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps,
    },
    repr::*,
    Element,
};
use std::marker::PhantomData;

macro_rules! make_ops {
    ($name:ident, $desc:ty, $fn:expr) => {
        #[derive(new)]
        struct $name<B: FusionBackend> {
            desc: $desc,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for $name<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                #[allow(clippy::redundant_closure_call)]
                $fn(self.desc, handles)
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
        make_ops!(
            Conv1dOps,
            Conv1dDescription,
            |desc: Conv1dDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&desc.x);
                let weight = handles.get_float_tensor::<B>(&desc.weight);
                let bias = desc
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));
                let output = B::conv1d(x, weight, bias, desc.options.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
        );

        let size = calculate_conv_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.dilation[0],
            x.shape[2],
        );

        let stream_1 = x.stream;
        let stream_2 = weight.stream;
        let stream_3 = bias.as_ref().map(|b| b.stream);
        let shape = vec![x.shape[0], weight.shape[0], size];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let description = Conv1dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match stream_3 {
            Some(stream_3) => vec![stream_1, stream_2, stream_3],
            None => vec![stream_1, stream_2],
        };
        out.client.clone().register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::Conv1d(description.clone())),
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
        make_ops!(
            Conv2dOps,
            Conv2dDescription,
            |args: Conv2dDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));

                let output = B::conv2d(x, weight, bias, args.options.clone().into());

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

        let stream_1 = x.stream;
        let stream_2 = weight.stream;
        let stream_3 = bias.as_ref().map(|b| b.stream);
        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = Conv2dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match stream_3 {
            Some(stream_3) => vec![stream_1, stream_2, stream_3],
            None => vec![stream_1, stream_2],
        };
        out.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::Conv2d(desc.clone())),
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
            DeformConv2dDescription,
            |args: DeformConv2dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = offset.stream;
        let stream_3 = weight.stream;
        let stream_4 = mask.as_ref().map(|m| m.stream);
        let stream_5 = bias.as_ref().map(|b| b.stream);
        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = DeformConv2dDescription {
            x: x.into_description(),
            offset: offset.into_description(),
            weight: weight.into_description(),
            mask: mask.map(|mask| mask.into_description()),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match (stream_4, stream_5) {
            (Some(stream_4), Some(stream_5)) => {
                vec![stream_1, stream_2, stream_3, stream_4, stream_5]
            }
            (Some(stream_4), None) => {
                vec![stream_1, stream_2, stream_3, stream_4]
            }
            (None, Some(stream_5)) => {
                vec![stream_1, stream_2, stream_3, stream_5]
            }
            (None, None) => vec![stream_1, stream_2, stream_3],
        };
        out.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::DeformableConv2d(Box::new(
                desc.clone(),
            ))),
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
            DeformConv2dBackwardDescription,
            |args: DeformConv2dBackwardDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = offset.stream;
        let stream_3 = weight.stream;
        let stream_4 = mask.as_ref().map(|m| m.stream);
        let stream_5 = bias.as_ref().map(|b| b.stream);
        let stream_6 = output_grad.stream;

        let desc = DeformConv2dBackwardDescription {
            x: x.into_description(),
            offset: offset.into_description(),
            weight: weight.into_description(),
            mask: mask.map(|mask| mask.into_description()),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out_grad: output_grad.into_description(),
            input_grad: input_grad.to_description_out(),
            offset_grad: offset_grad.to_description_out(),
            weight_grad: weight_grad.to_description_out(),
            mask_grad: mask_grad
                .as_ref()
                .map(|mask_grad| mask_grad.to_description_out()),
            bias_grad: bias_grad
                .as_ref()
                .map(|bias_grad| bias_grad.to_description_out()),
        };

        let streams = match (stream_4, stream_5) {
            (Some(stream_4), Some(stream_5)) => {
                vec![stream_1, stream_2, stream_3, stream_4, stream_5, stream_6]
            }
            (Some(stream_4), None) => {
                vec![stream_1, stream_2, stream_3, stream_4, stream_6]
            }
            (None, Some(stream_5)) => {
                vec![stream_1, stream_2, stream_3, stream_5, stream_6]
            }
            (None, None) => vec![stream_1, stream_2, stream_3, stream_6],
        };

        input_grad.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::DeformableConv2dBackward(
                Box::new(desc.clone()),
            )),
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
        make_ops!(
            Conv3dOps,
            Conv3dDescription,
            |args: Conv3dDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let weight = handles.get_float_tensor::<B>(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor::<B>(bias));

                let output = B::conv3d(x, weight, bias, args.options.clone().into());

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
        let size_2 = calculate_conv_output_size(
            weight.shape[4],
            options.stride[2],
            options.padding[2],
            options.dilation[2],
            x.shape[4],
        );

        let stream_1 = x.stream;
        let stream_2 = weight.stream;
        let stream_3 = bias.as_ref().map(|b| b.stream);
        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1, size_2];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = Conv3dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match stream_3 {
            Some(stream_3) => vec![stream_1, stream_2, stream_3],
            None => vec![stream_1, stream_2],
        };
        out.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::Conv3d(desc.clone())),
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
            ConvTranspose1dDescription,
            |args: ConvTranspose1dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = weight.stream;
        let stream_3 = bias.as_ref().map(|b| b.stream);
        let shape = vec![x.shape[0], weight.shape[1] * options.groups, size];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ConvTranspose1dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match stream_3 {
            Some(stream_3) => vec![stream_1, stream_2, stream_3],
            None => vec![stream_1, stream_2],
        };
        out.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::ConvTranspose1d(desc.clone())),
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
            ConvTranspose2dDescription,
            |args: ConvTranspose2dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = weight.stream;
        let stream_3 = bias.as_ref().map(|b| b.stream);
        let shape = vec![x.shape[0], weight.shape[1] * options.groups, size_0, size_1];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ConvTranspose2dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match stream_3 {
            Some(stream_3) => vec![stream_1, stream_2, stream_3],
            None => vec![stream_1, stream_2],
        };
        out.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::ConvTranspose2d(desc.clone())),
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
            ConvTranspose3dDescription,
            |args: ConvTranspose3dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = weight.stream;
        let stream_3 = bias.as_ref().map(|b| b.stream);
        let shape = vec![
            x.shape[0],
            weight.shape[1] * options.groups,
            size_0,
            size_1,
            size_2,
        ];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ConvTranspose3dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options: options.into(),
            out: out.to_description_out(),
        };

        let streams = match stream_3 {
            Some(stream_3) => vec![stream_1, stream_2, stream_3],
            None => vec![stream_1, stream_2],
        };
        out.client.register(
            streams,
            OperationDescription::Module(ModuleOperationDescription::ConvTranspose3d(desc.clone())),
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
            AvgPool1dDescription,
            |args: AvgPool1dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream = x.stream;
        let size = calculate_pool_output_size(kernel_size, stride, padding, 1, x.shape[2]);
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = AvgPool1dDescription {
            x: x.into_description(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::AvgPool1d(desc.clone())),
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
            AvgPool2dDescription,
            |args: AvgPool2dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = AvgPool2dDescription {
            x: x.into_description(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::AvgPool2d(desc.clone())),
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
            AvgPool1dBackwardDescription,
            |args: AvgPool1dBackwardDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = grad.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = AvgPool1dBackwardDescription {
            x: x.into_description(),
            grad: grad.into_description(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Module(ModuleOperationDescription::AvgPool1dBackward(
                desc.clone(),
            )),
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
            AvgPool2dBackwardDescription,
            |args: AvgPool2dBackwardDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = grad.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = AvgPool2dBackwardDescription {
            x: x.into_description(),
            grad: grad.into_description(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Module(ModuleOperationDescription::AvgPool2dBackward(
                desc.clone(),
            )),
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
            MaxPool1dDescription,
            |args: MaxPool1dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = MaxPool1dDescription {
            x: x.into_description(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::MaxPool1d(desc.clone())),
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
            MaxPool2dDescription,
            |args: MaxPool2dDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = MaxPool2dDescription {
            x: x.into_description(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::MaxPool2d(desc.clone())),
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
            MaxPool1dWithIndicesDescription,
            |args: MaxPool1dWithIndicesDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream = x.stream;
        let size = calculate_pool_output_size(kernel_size, stride, padding, dilation, x.shape[2]);
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x
            .client
            .tensor_uninitialized(shape.clone(), B::FloatElem::dtype());
        let out_indices = x.client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = MaxPool1dWithIndicesDescription {
            x: x.into_description(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::MaxPool1dWithIndices(
                desc.clone(),
            )),
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
            MaxPool2dWithIndicesDescription,
            |args: MaxPool2dWithIndicesDescription, handles: &mut HandleContainer<B::Handle>| {
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

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x
            .client
            .tensor_uninitialized(shape.clone(), B::FloatElem::dtype());
        let out_indices = x.client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = MaxPool2dWithIndicesDescription {
            x: x.into_description(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::MaxPool2dWithIndices(
                desc.clone(),
            )),
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
            MaxPool1dWithIndicesBackwardDescription,
            |args: MaxPool1dWithIndicesBackwardDescription,
             handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = output_grad.stream;
        let stream_3 = indices.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = MaxPool1dWithIndicesBackwardDescription {
            x: x.into_description(),
            grad: output_grad.into_description(),
            indices: indices.into_description(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::Module(ModuleOperationDescription::MaxPool1dWithIndicesBackward(
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
            MaxPool2dWithIndicesBackwardDescription,
            |args: MaxPool2dWithIndicesBackwardDescription,
             handles: &mut HandleContainer<B::Handle>| {
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

        let stream_1 = x.stream;
        let stream_2 = output_grad.stream;
        let stream_3 = indices.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = MaxPool2dWithIndicesBackwardDescription {
            x: x.into_description(),
            grad: output_grad.into_description(),
            indices: indices.into_description(),
            kernel_size,
            stride,
            padding,
            dilation,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::Module(ModuleOperationDescription::MaxPool2dWithIndicesBackward(
                desc.clone(),
            )),
            MaxPool2dWithIndicesBackwardOps::<B>::new(desc),
        );

        MaxPool2dBackward::new(out)
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self>, output_size: usize) -> FloatTensor<Self> {
        make_ops!(
            AdaptiveAvgPool1dOps,
            AdaptiveAvgPool1dDescription,
            |args: AdaptiveAvgPool1dDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::adaptive_avg_pool1d(x, args.output_size);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], output_size];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = AdaptiveAvgPool1dDescription {
            x: x.into_description(),
            output_size,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::AdaptiveAvgPool1d(
                desc.clone(),
            )),
            AdaptiveAvgPool1dOps::<B>::new(desc),
        );

        out
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        make_ops!(
            AdaptiveAvgPool2dOps,
            AdaptiveAvgPool2dDescription,
            |args: AdaptiveAvgPool2dDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::adaptive_avg_pool2d(x, args.output_size);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], output_size[0], output_size[1]];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = AdaptiveAvgPool2dDescription {
            x: x.into_description(),
            output_size,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::AdaptiveAvgPool2d(
                desc.clone(),
            )),
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
            AdaptiveAvgPool1dBackwardDescription,
            |args: AdaptiveAvgPool1dBackwardDescription,
             handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output = B::adaptive_avg_pool1d_backward(x, grad);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let stream_1 = x.stream;
        let stream_2 = grad.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());
        let desc = AdaptiveAvgPool1dBackwardDescription {
            x: x.into_description(),
            grad: grad.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Module(ModuleOperationDescription::AdaptiveAvgPool1dBackward(
                desc.clone(),
            )),
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
            AdaptiveAvgPool2dBackwardDescription,
            |args: AdaptiveAvgPool2dBackwardDescription,
             handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output = B::adaptive_avg_pool2d_backward(x, grad);

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let stream_1 = x.stream;
        let stream_2 = grad.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = AdaptiveAvgPool2dBackwardDescription {
            x: x.into_description(),
            grad: grad.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Module(ModuleOperationDescription::AdaptiveAvgPool2dBackward(
                desc.clone(),
            )),
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
            InterpolateDescription,
            |args: InterpolateDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let output = B::interpolate(x, args.output_size, args.options.clone().into());
                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let stream = x.stream;
        let shape = vec![x.shape[0], x.shape[1], output_size[0], output_size[1]];
        let out = x.client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = InterpolateDescription {
            x: x.into_description(),
            output_size,
            options: options.into(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::Module(ModuleOperationDescription::Interpolate(desc.clone())),
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
            InterpolateBackwardDescription,
            |args: InterpolateBackwardDescription, handles: &mut HandleContainer<B::Handle>| {
                let x = handles.get_float_tensor::<B>(&args.x);
                let grad = handles.get_float_tensor::<B>(&args.grad);
                let output =
                    B::interpolate_backward(x, grad, args.output_size, args.options.clone().into());

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let stream_1 = x.stream;
        let stream_2 = grad.stream;
        let out = x
            .client
            .tensor_uninitialized(x.shape.clone(), B::FloatElem::dtype());

        let desc = InterpolateBackwardDescription {
            x: x.into_description(),
            grad: grad.into_description(),
            output_size,
            options: options.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Module(ModuleOperationDescription::InterpolateBackward(
                desc.clone(),
            )),
            InterpolateBackwardOps::<B>::new(desc),
        );
        out
    }
}
