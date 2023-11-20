use crate::{
    client::FusionClient,
    graph::{
        AdaptiveAvgPool1dBackwardDescription, AdaptiveAvgPool1dDescription,
        AdaptiveAvgPool2dBackwardDescription, AdaptiveAvgPool2dDescription,
        AvgPool1dBackwardDescription, AvgPool1dDescription, AvgPool2dBackwardDescription,
        AvgPool2dDescription, Conv1dDescription, Conv2dDescription, ConvTranspose1dDescription,
        ConvTranspose2dDescription, MaxPool1dDescription, MaxPool1dWithIndicesBackwardDescription,
        MaxPool1dWithIndicesDescription, MaxPool2dDescription,
        MaxPool2dWithIndicesBackwardDescription, MaxPool2dWithIndicesDescription, Ops,
        TensorOpsDescription,
    },
    Fusion, FusionBackend,
};
use burn_tensor::ops::{
    conv::{
        calculate_conv_output_size, calculate_conv_transpose_output_size,
        calculate_pool_output_size,
    },
    ConvOptions, ConvTransposeOptions, FloatTensor, IntTensor, MaxPool1dBackward,
    MaxPool1dWithIndices, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};

macro_rules! make_ops {
    ($name:ident, $desc:ty, $fn:expr) => {
        struct $name {
            desc: $desc,
        }

        impl<B: FusionBackend> Ops<B> for $name {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                $fn(self.desc, handles)
            }
        }
    };
}

impl<B: FusionBackend> ModuleOps<Fusion<B>> for Fusion<B> {
    fn conv1d(
        x: FloatTensor<Self, 3>,
        weight: FloatTensor<Self, 3>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<Self, 3> {
        make_ops!(Conv1dOps, Conv1dDescription, |desc, handles| {
            let x = handles.get_float_tensor(&desc.x);
            let weight = handles.get_float_tensor(&desc.weight);
            let bias = desc
                .bias
                .as_ref()
                .map(|bias| handles.get_float_tensor(bias));
            let output = B::conv1d(x, weight, bias, desc.options);
            handles.register_float_tensor(&desc.out.id, output);
        });

        let size = calculate_conv_output_size(
            weight.shape[2],
            options.stride[0],
            options.padding[0],
            options.dilation[0],
            x.shape[2],
        );

        let shape = vec![x.shape[0], weight.shape[0], size];
        let out = x.client.tensor_uninitialized(shape);

        let description = Conv1dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options,
            out: out.to_description_out(),
        };
        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::Conv1d(
                description.clone(),
            )),
            Conv1dOps::new(description),
        );

        out
    }

    fn conv2d(
        x: FloatTensor<Self, 4>,
        weight: FloatTensor<Self, 4>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self, 4> {
        make_ops!(Conv2dOps, Conv2dDescription, |args, handles| {
            let x = handles.get_float_tensor(&args.x);
            let weight = handles.get_float_tensor(&args.weight);
            let bias = args
                .bias
                .as_ref()
                .map(|bias| handles.get_float_tensor(bias));

            let output = B::conv2d(x, weight, bias, args.options.clone());

            handles.register_float_tensor(&args.out.id, output);
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

        let shape = vec![x.shape[0], weight.shape[0], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape);

        let desc = Conv2dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options,
            out: out.to_description_out(),
        };
        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::Conv2d(
                desc.clone(),
            )),
            Box::new(Conv2dOps::new(desc)),
        );

        out
    }

    fn conv_transpose1d(
        x: FloatTensor<Self, 3>,
        weight: FloatTensor<Self, 3>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<Self, 3> {
        make_ops!(
            ConvTranspose1dOps,
            ConvTranspose1dDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let weight = handles.get_float_tensor(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor(bias));

                let output = B::conv_transpose1d(x, weight, bias, args.options.clone());

                handles.register_float_tensor(&args.out.id, output);
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

        let shape = vec![x.shape[0], weight.shape[1] * options.groups, size];
        let out = x.client.tensor_uninitialized(shape);

        let desc = ConvTranspose1dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options,
            out: out.to_description_out(),
        };
        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::ConvTranspose1d(
                desc.clone(),
            )),
            ConvTranspose1dOps::new(desc),
        );

        out
    }

    fn conv_transpose2d(
        x: FloatTensor<Self, 4>,
        weight: FloatTensor<Self, 4>,
        bias: Option<FloatTensor<Self, 1>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self, 4> {
        make_ops!(
            ConvTranspose2dOps,
            ConvTranspose2dDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let weight = handles.get_float_tensor(&args.weight);
                let bias = args
                    .bias
                    .as_ref()
                    .map(|bias| handles.get_float_tensor(bias));

                let output = B::conv_transpose2d(x, weight, bias, args.options.clone());

                handles.register_float_tensor(&args.out.id, output);
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

        let shape = vec![x.shape[0], weight.shape[1] * options.groups, size_0, size_1];
        let out = x.client.tensor_uninitialized(shape);

        let desc = ConvTranspose2dDescription {
            x: x.into_description(),
            weight: weight.into_description(),
            bias: bias.map(|bias| bias.into_description()),
            options,
            out: out.to_description_out(),
        };
        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::ConvTranspose2d(
                desc.clone(),
            )),
            ConvTranspose2dOps::new(desc),
        );

        out
    }

    fn avg_pool1d(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self, 3> {
        make_ops!(AvgPool1dOps, AvgPool1dDescription, |args, handles| {
            let x = handles.get_float_tensor(&args.x);
            let output = B::avg_pool1d(
                x,
                args.kernel_size,
                args.stride,
                args.padding,
                args.count_include_pad,
            );

            handles.register_float_tensor(&args.out.id, output);
        });

        let size = calculate_pool_output_size(kernel_size, stride, padding, 1, x.shape[2]);
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::AvgPool1d(
                AvgPool1dDescription {
                    x: x.into_description(),
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    out: out.to_description_out(),
                },
            )),
            AvgPool1dOps::new(todo!()),
        );

        out
    }

    fn avg_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self, 4> {
        make_ops!(AvgPool2dOps, AvgPool2dDescription, |args, handles| {
            let x = handles.get_float_tensor(&args.x);
            let output = B::avg_pool2d(
                x,
                args.kernel_size,
                args.stride,
                args.padding,
                args.count_include_pad,
            );

            handles.register_float_tensor(&args.out.id, output);
        });

        let size_0 =
            calculate_pool_output_size(kernel_size[0], stride[0], padding[0], 1, x.shape[2]);
        let size_1 =
            calculate_pool_output_size(kernel_size[1], stride[1], padding[1], 1, x.shape[3]);

        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::AvgPool2d(
                AvgPool2dDescription {
                    x: x.into_description(),
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn avg_pool1d_backward(
        x: FloatTensor<Self, 3>,
        grad: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
    ) -> FloatTensor<Self, 3> {
        make_ops!(
            AvgPool1dBackwardOps,
            AvgPool1dBackwardDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let grad = handles.get_float_tensor(&args.grad);
                let output = B::avg_pool1d_backward(
                    x,
                    grad,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.count_include_pad,
                );

                handles.register_float_tensor(&args.out.id, output);
            }
        );

        let out = x.client.tensor_uninitialized(x.shape.clone());

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::AvgPool1dBackward(
                AvgPool1dBackwardDescription {
                    x: x.into_description(),
                    grad: grad.into_description(),
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> FloatTensor<Self, 4> {
        make_ops!(
            AvgPool2dBackwardOps,
            AvgPool2dBackwardDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let grad = handles.get_float_tensor(&args.grad);
                let output = B::avg_pool2d_backward(
                    x,
                    grad,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.count_include_pad,
                );

                handles.register_float_tensor(&args.out.id, output);
            }
        );

        let out = x.client.tensor_uninitialized(x.shape.clone());

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::AvgPool2dBackward(
                AvgPool2dBackwardDescription {
                    x: x.into_description(),
                    grad: grad.into_description(),
                    kernel_size,
                    stride,
                    padding,
                    count_include_pad,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn max_pool1d(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> FloatTensor<Self, 3> {
        make_ops!(MaxPool1dOps, MaxPool1dDescription, |args, handles| {
            let x = handles.get_float_tensor(&args.x);
            let output = B::max_pool1d(
                x,
                args.kernel_size,
                args.stride,
                args.padding,
                args.dilation,
            );

            handles.register_float_tensor(&args.out.id, output);
        });

        let size = calculate_pool_output_size(kernel_size, stride, padding, dilation, x.shape[2]);

        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::MaxPool1d(
                MaxPool1dDescription {
                    x: x.into_description(),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn max_pool2d(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        make_ops!(MaxPool2dOps, MaxPool2dDescription, |args, handles| {
            let x = handles.get_float_tensor(&args.x);
            let output = B::max_pool2d(
                x,
                args.kernel_size,
                args.stride,
                args.padding,
                args.dilation,
            );

            handles.register_float_tensor(&args.out.id, output);
        });

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

        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::MaxPool2d(
                MaxPool2dDescription {
                    x: x.into_description(),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn max_pool1d_with_indices(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<Self> {
        make_ops!(
            MaxPool1dWithIndicesOps,
            MaxPool1dWithIndicesDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let output = B::max_pool1d_with_indices(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                );

                handles.register_float_tensor(&args.out.id, output.output);
                handles.register_int_tensor(&args.out_indices.id, output.indices);
            }
        );

        let size = calculate_pool_output_size(kernel_size, stride, padding, dilation, x.shape[2]);
        let shape = vec![x.shape[0], x.shape[1], size];
        let out = x.client.tensor_uninitialized(shape.clone());
        let out_indices = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(
                crate::graph::ModuleOpsDescription::MaxPool1dWithIndices(
                    MaxPool1dWithIndicesDescription {
                        x: x.into_description(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        out: out.to_description_out(),
                        out_indices: out_indices.to_description_out(),
                    },
                ),
            ),
            todo!(),
        );

        MaxPool1dWithIndices::new(out, out_indices)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Self> {
        make_ops!(
            MaxPool2dWithIndicesOps,
            MaxPool2dWithIndicesDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let output = B::max_pool2d_with_indices(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                );

                handles.register_float_tensor(&args.out.id, output.output);
                handles.register_int_tensor(&args.out_indices.id, output.indices);
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

        let shape = vec![x.shape[0], x.shape[1], size_0, size_1];
        let out = x.client.tensor_uninitialized(shape.clone());
        let out_indices = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(
                crate::graph::ModuleOpsDescription::MaxPool2dWithIndices(
                    MaxPool2dWithIndicesDescription {
                        x: x.into_description(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        out: out.to_description_out(),
                        out_indices: out_indices.to_description_out(),
                    },
                ),
            ),
            todo!(),
        );

        MaxPool2dWithIndices::new(out, out_indices)
    }

    fn max_pool1d_with_indices_backward(
        x: FloatTensor<Self, 3>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_grad: FloatTensor<Self, 3>,
        indices: IntTensor<Self, 3>,
    ) -> MaxPool1dBackward<Self> {
        make_ops!(
            MaxPool1dWithIndicesBackwardOps,
            MaxPool1dWithIndicesBackwardDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let grad = handles.get_float_tensor(&args.grad);
                let indices = handles.get_int_tensor(&args.indices);
                let output = B::max_pool1d_with_indices_backward(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                    grad,
                    indices,
                );

                handles.register_float_tensor(&args.out.id, output.x_grad);
            }
        );

        let out = x.client.tensor_uninitialized(x.shape.clone());

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(
                crate::graph::ModuleOpsDescription::MaxPool1dWithIndicesBackward(
                    MaxPool1dWithIndicesBackwardDescription {
                        x: x.into_description(),
                        grad: output_grad.into_description(),
                        indices: indices.into_description(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        out: out.to_description_out(),
                    },
                ),
            ),
            todo!(),
        );

        MaxPool1dBackward::new(out)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self, 4>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: FloatTensor<Self, 4>,
        indices: IntTensor<Self, 4>,
    ) -> MaxPool2dBackward<Self> {
        make_ops!(
            MaxPool2dWithIndicesBackwardOps,
            MaxPool2dWithIndicesBackwardDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let grad = handles.get_float_tensor(&args.grad);
                let indices = handles.get_int_tensor(&args.indices);
                let output = B::max_pool2d_with_indices_backward(
                    x,
                    args.kernel_size,
                    args.stride,
                    args.padding,
                    args.dilation,
                    grad,
                    indices,
                );

                handles.register_float_tensor(&args.out.id, output.x_grad);
            }
        );

        let out = x.client.tensor_uninitialized(x.shape.clone());

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(
                crate::graph::ModuleOpsDescription::MaxPool2dWithIndicesBackward(
                    MaxPool2dWithIndicesBackwardDescription {
                        x: x.into_description(),
                        grad: output_grad.into_description(),
                        indices: indices.into_description(),
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        out: out.to_description_out(),
                    },
                ),
            ),
            todo!(),
        );

        MaxPool2dBackward::new(out)
    }

    fn adaptive_avg_pool1d(x: FloatTensor<Self, 3>, output_size: usize) -> FloatTensor<Self, 3> {
        make_ops!(
            AdaptiveAvgPool1dOps,
            AdaptiveAvgPool1dDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let output = B::adaptive_avg_pool1d(x, args.output_size);

                handles.register_float_tensor(&args.out.id, output);
            }
        );

        let shape = vec![x.shape[0], x.shape[1], output_size];
        let out = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::AdaptiveAvgPool1d(
                AdaptiveAvgPool1dDescription {
                    x: x.into_description(),
                    output_size,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn adaptive_avg_pool2d(
        x: FloatTensor<Self, 4>,
        output_size: [usize; 2],
    ) -> FloatTensor<Self, 4> {
        make_ops!(
            AdaptiveAvgPool2dOps,
            AdaptiveAvgPool2dDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let output = B::adaptive_avg_pool2d(x, args.output_size);

                handles.register_float_tensor(&args.out.id, output);
            }
        );

        let shape = vec![x.shape[0], x.shape[1], output_size[0], output_size[1]];
        let out = x.client.tensor_uninitialized(shape);

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(crate::graph::ModuleOpsDescription::AdaptiveAvgPool2d(
                AdaptiveAvgPool2dDescription {
                    x: x.into_description(),
                    output_size,
                    out: out.to_description_out(),
                },
            )),
            todo!(),
        );

        out
    }

    fn adaptive_avg_pool1d_backward(
        x: FloatTensor<Self, 3>,
        grad: FloatTensor<Self, 3>,
    ) -> FloatTensor<Self, 3> {
        make_ops!(
            AdaptiveAvgPool1dBackwardOps,
            AdaptiveAvgPool1dBackwardDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let grad = handles.get_float_tensor(&args.grad);
                let output = B::adaptive_avg_pool1d_backward(x, grad);

                handles.register_float_tensor(&args.out.id, output);
            }
        );

        let out = x.client.tensor_uninitialized(x.shape.clone());

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(
                crate::graph::ModuleOpsDescription::AdaptiveAvgPool1dBackward(
                    AdaptiveAvgPool1dBackwardDescription {
                        x: x.into_description(),
                        grad: grad.into_description(),
                        out: out.to_description_out(),
                    },
                ),
            ),
            todo!(),
        );

        out
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self, 4>,
        grad: FloatTensor<Self, 4>,
    ) -> FloatTensor<Self, 4> {
        make_ops!(
            AdaptiveAvgPool2dBackwardOps,
            AdaptiveAvgPool2dBackwardDescription,
            |args, handles| {
                let x = handles.get_float_tensor(&args.x);
                let grad = handles.get_float_tensor(&args.grad);
                let output = B::adaptive_avg_pool2d_backward(x, grad);

                handles.register_float_tensor(&args.out.id, output);
            }
        );

        let out = x.client.tensor_uninitialized(x.shape.clone());

        x.client.clone().register(
            TensorOpsDescription::ModuleOps(
                crate::graph::ModuleOpsDescription::AdaptiveAvgPool2dBackward(
                    AdaptiveAvgPool2dBackwardDescription {
                        x: x.into_description(),
                        grad: grad.into_description(),
                        out: out.to_description_out(),
                    },
                ),
            ),
            todo!(),
        );

        out
    }
}
