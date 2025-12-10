use crate::{
    Fusion, FusionBackend,
    stream::{OperationStreams, execution::Operation},
};
use burn_ir::*;
use burn_tensor::{
    Element,
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions, FloatTensor,
        IntTensor, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
        MaxPool2dWithIndices, ModuleOps,
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

        let mut streams = OperationStreams::with_inputs([&x, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let client = x.client.clone();
        let desc = Conv1dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::Conv1d(desc.clone())),
                Conv1dOps::<B>::new(desc),
            )
            .output()
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

        let mut streams = OperationStreams::with_inputs([&x, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let client = x.client.clone();
        let desc = Conv2dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::Conv2d(desc.clone())),
                Conv2dOps::<B>::new(desc),
            )
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
        let mut streams = OperationStreams::with_inputs([&x, &offset, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }
        if let Some(mask) = mask.as_ref() {
            streams.tensor(mask)
        }

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
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::DeformableConv2d(Box::new(desc.clone()))),
                DeformConv2dOps::<B>::new(desc),
            )
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

        let has_bias = bias.is_some();
        let has_mask = mask.is_some();

        let mut streams = OperationStreams::with_inputs([&x, &offset, &weight, &output_grad]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias);
        }
        if let Some(mask) = mask.as_ref() {
            streams.tensor(mask);
        }

        let client = x.client.clone();
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
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::DeformableConv2dBackward(Box::new(
                    desc.clone(),
                ))),
                DeformConv2dBackwardOps::<B>::new(desc),
            )
            .into_iter();

        // When the number of outputs is variable, the order is important
        let input_grad = outputs.next().unwrap();
        let offset_grad = outputs.next().unwrap();
        let weight_grad = outputs.next().unwrap();
        let mask_grad = has_mask.then(|| outputs.next().unwrap());
        let bias_grad = has_bias.then(|| outputs.next().unwrap());

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

        let mut streams = OperationStreams::with_inputs([&x, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let client = x.client.clone();
        let desc = Conv3dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::Conv3d(desc.clone())),
                Conv3dOps::<B>::new(desc),
            )
            .output()
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
        let mut streams = OperationStreams::with_inputs([&x, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let client = x.client.clone();
        let desc = ConvTranspose1dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::ConvTranspose1d(desc.clone())),
                ConvTranspose1dOps::<B>::new(desc),
            )
            .output()
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
        let mut streams = OperationStreams::with_inputs([&x, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let client = x.client.clone();
        let desc = ConvTranspose2dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::ConvTranspose2d(desc.clone())),
                ConvTranspose2dOps::<B>::new(desc),
            )
            .output()
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
        let mut streams = OperationStreams::with_inputs([&x, &weight]);
        if let Some(bias) = bias.as_ref() {
            streams.tensor(bias)
        }

        let client = x.client.clone();
        let desc = ConvTranspose3dOpIr::create(
            x.into_ir(),
            weight.into_ir(),
            bias.map(|bias| bias.into_ir()),
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::ConvTranspose3d(desc.clone())),
                ConvTranspose3dOps::<B>::new(desc),
            )
            .output()
    }

    fn avg_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );
        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = AvgPool1dOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AvgPool1d(desc.clone())),
                AvgPool1dOps::<B>::new(desc),
            )
            .output()
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = AvgPool2dOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AvgPool2d(desc.clone())),
                AvgPool2dOps::<B>::new(desc),
            )
            .output()
    }

    fn avg_pool1d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let streams = OperationStreams::with_inputs([&x, &grad]);

        let client = x.client.clone();
        let desc = AvgPool1dBackwardOpIr::create(
            x.into_ir(),
            grad.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AvgPool1dBackward(desc.clone())),
                AvgPool1dBackwardOps::<B>::new(desc),
            )
            .output()
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let streams = OperationStreams::with_inputs([&x, &grad]);

        let client = x.client.clone();
        let desc = AvgPool2dBackwardOpIr::create(
            x.into_ir(),
            grad.into_ir(),
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AvgPool2dBackward(desc.clone())),
                AvgPool2dBackwardOps::<B>::new(desc),
            )
            .output()
    }

    fn max_pool1d(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = MaxPool1dOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::MaxPool1d(desc.clone())),
                MaxPool1dOps::<B>::new(desc),
            )
            .output()
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output);
            }
        );

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = MaxPool2dOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::MaxPool2d(desc.clone())),
                MaxPool2dOps::<B>::new(desc),
            )
            .output()
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.output);
                handles.register_int_tensor::<B>(&args.out_indices.id, output.indices);
            }
        );

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = MaxPool1dWithIndicesOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            B::IntElem::dtype(),
            || client.create_empty_handle(),
        );

        let [out, out_indices] = client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::MaxPool1dWithIndices(desc.clone())),
                MaxPool1dWithIndicesOps::<B>::new(desc),
            )
            .outputs();

        MaxPool1dWithIndices::new(out, out_indices)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
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
                    args.ceil_mode,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.output);
                handles.register_int_tensor::<B>(&args.out_indices.id, output.indices);
            }
        );

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = MaxPool2dWithIndicesOpIr::create(
            x.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            B::IntElem::dtype(),
            || client.create_empty_handle(),
        );

        let [out, out_indices] = client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::MaxPool2dWithIndices(desc.clone())),
                MaxPool2dWithIndicesOps::<B>::new(desc),
            )
            .outputs();

        MaxPool2dWithIndices::new(out, out_indices)
    }

    fn max_pool1d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
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
                    args.ceil_mode,
                    grad,
                    indices,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.x_grad);
            }
        );

        let streams = OperationStreams::with_inputs([&x, &output_grad, &indices]);

        let client = x.client.clone();
        let desc = MaxPool1dWithIndicesBackwardOpIr::create(
            x.into_ir(),
            output_grad.into_ir(),
            indices.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            || client.create_empty_handle(),
        );

        let out = client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::MaxPool1dWithIndicesBackward(
                    desc.clone(),
                )),
                MaxPool1dWithIndicesBackwardOps::<B>::new(desc),
            )
            .output();

        MaxPool1dBackward::new(out)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
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
                    args.ceil_mode,
                    grad,
                    indices,
                );

                handles.register_float_tensor::<B>(&args.out.id, output.x_grad);
            }
        );

        let streams = OperationStreams::with_inputs([&x, &output_grad, &indices]);

        let client = x.client.clone();
        let desc = MaxPool2dWithIndicesBackwardOpIr::create(
            x.into_ir(),
            output_grad.into_ir(),
            indices.into_ir(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            || client.create_empty_handle(),
        );

        let out = client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::MaxPool2dWithIndicesBackward(
                    desc.clone(),
                )),
                MaxPool2dWithIndicesBackwardOps::<B>::new(desc),
            )
            .output();

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

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = AdaptiveAvgPool1dOpIr::create(x.into_ir(), output_size, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool1d(desc.clone())),
                AdaptiveAvgPool1dOps::<B>::new(desc),
            )
            .output()
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

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = AdaptiveAvgPool2dOpIr::create(x.into_ir(), output_size, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2d(desc.clone())),
                AdaptiveAvgPool2dOps::<B>::new(desc),
            )
            .output()
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

        let streams = OperationStreams::with_inputs([&x, &grad]);

        let client = x.client.clone();
        let desc = AdaptiveAvgPool1dBackwardOpIr::create(x.into_ir(), grad.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool1dBackward(desc.clone())),
                AdaptiveAvgPool1dBackwardOps::<B>::new(desc),
            )
            .output()
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
        let streams = OperationStreams::with_inputs([&x, &grad]);

        let client = x.client.clone();
        let desc = AdaptiveAvgPool2dBackwardOpIr::create(x.into_ir(), grad.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::AdaptiveAvgPool2dBackward(desc.clone())),
                AdaptiveAvgPool2dBackwardOps::<B>::new(desc),
            )
            .output()
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

        let streams = OperationStreams::with_inputs([&x]);

        let client = x.client.clone();
        let desc = InterpolateOpIr::create(x.into_ir(), output_size, options.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::Interpolate(desc.clone())),
                InterpolateOps::<B>::new(desc),
            )
            .output()
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

        let streams = OperationStreams::with_inputs([&x, &grad]);

        let client = x.client.clone();
        let desc = InterpolateBackwardOpIr::create(
            x.into_ir(),
            grad.into_ir(),
            output_size,
            options.into(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::Module(ModuleOperationIr::InterpolateBackward(desc.clone())),
                InterpolateBackwardOps::<B>::new(desc),
            )
            .output()
    }
}
