use burn_cube::{calculate_cube_count_elemwise, prelude::*, SUBCUBE_DIM_APPROX};

use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

use crate::{
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn kernel<F: Float>(
    input: Tensor<F>,
    weight: Tensor<F>,
    bias: Tensor<F>,
    mut output: Tensor<F>,
    conv_stride_0: UInt,
    conv_stride_1: UInt,
    dilation_0: UInt,
    dilation_1: UInt,
    padding_0: UInt,
    padding_1: UInt,
    groups: UInt,
    kernel_size_0_unroll: Comptime<Option<UInt>>,
    kernel_size_1_unroll: Comptime<Option<UInt>>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }
    let in_channels = weight.shape(1);
    let kernel_size_0 = Comptime::unwrap_or_else(kernel_size_0_unroll, || weight.shape(2));
    let unroll_0 = Comptime::is_some(kernel_size_0_unroll);
    let kernel_size_1 = Comptime::unwrap_or_else(kernel_size_1_unroll, || weight.shape(3));
    let unroll_1 = Comptime::is_some(kernel_size_1_unroll);
    let b = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let oc = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let oh = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let ow = ABSOLUTE_POS / output.stride(3) % output.shape(3);
    let g = (weight.shape(0) + oc) % groups;
    let ic_start = in_channels * g;
    let ic_end = ic_start + in_channels;
    let mut sum = bias[oc];
    let ih_base = oh * conv_stride_0;
    let iw_base = ow * conv_stride_1;
    let weight_stride_1 = weight.stride(1);
    let weight_stride_2 = weight.stride(2);
    let weight_stride_3 = weight.stride(3);
    let input_stride_1 = input.stride(1);
    let input_stride_2 = input.stride(2);
    let input_stride_3 = input.stride(3);
    let input_shape_2 = input.shape(2);
    let input_shape_3 = input.shape(3);
    let border_top = padding_0;
    let border_left = padding_1;
    let border_bottom = input_shape_2 + padding_0;
    let border_right = input_shape_3 + padding_1;
    let index_input_0 = b * input.stride(0);
    let index_weight_0 = oc * weight.stride(0);
    for ic in range(ic_start, ic_end, Comptime::new(false)) {
        let index_input_1 = ic * input_stride_1;
        let index_weight_1 = (ic - ic_start) * weight_stride_1;
        for kh in range(0, kernel_size_0, unroll_0) {
            for kw in range(0, kernel_size_1, unroll_1) {
                let ih = kh * dilation_0 + ih_base;
                let iw = kw * dilation_1 + iw_base;
                let within_padding = ih >= border_top
                    && ih < border_bottom
                    && iw >= border_left
                    && iw < border_right;
                if within_padding {
                    let ih_pad = ih - padding_0;
                    let iw_pad = iw - padding_1;
                    let index_input = index_input_0
                        + index_input_1
                        + ih_pad * input_stride_2
                        + iw_pad * input_stride_3;
                    let index_weight = index_weight_0
                        + index_weight_1
                        + kh * weight_stride_2
                        + kw * weight_stride_3;
                    sum += input[index_input] * weight[index_weight];
                }
            }
        }
    }
    output[ABSOLUTE_POS] = sum;
}
#[allow(unused_mut)]
#[allow(clippy::too_many_arguments)]
pub fn kernel_expand<F: Float>(
    context: &mut burn_cube::frontend::CubeContext,
    input: <Tensor<F> as burn_cube::frontend::CubeType>::ExpandType,
    weight: <Tensor<F> as burn_cube::frontend::CubeType>::ExpandType,
    bias: <Tensor<F> as burn_cube::frontend::CubeType>::ExpandType,
    mut output: <Tensor<F> as burn_cube::frontend::CubeType>::ExpandType,
    conv_stride_0: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    conv_stride_1: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    dilation_0: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    dilation_1: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    padding_0: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    padding_1: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    groups: <UInt as burn_cube::frontend::CubeType>::ExpandType,
    kernel_size_0_unroll: <Comptime<Option<UInt>> as burn_cube::frontend::CubeType>::ExpandType,
    kernel_size_1_unroll: <Comptime<Option<UInt>> as burn_cube::frontend::CubeType>::ExpandType,
) -> () {
    let _cond = {
        let _lhs = ABSOLUTE_POS::expand(context);
        let _rhs = output.clone().len_expand(context);
        burn_cube::frontend::ge::expand(context, _lhs, _rhs)
    };
    burn_cube::frontend::branch::if_expand(context, None, _cond.into(), |context| {
        burn_cube::frontend::branch::return_expand(context);
    });
    let in_channels = weight.clone().shape_expand(context, 1);
    let kernel_size_0 =
        Comptime::unwrap_or_else_expand(context, kernel_size_0_unroll.clone(), |context| {
            weight.clone().shape_expand(context, 2)
        });
    let unroll_0 = kernel_size_0_unroll.is_some();
    let kernel_size_1 =
        Comptime::unwrap_or_else_expand(context, kernel_size_1_unroll.clone(), |context| {
            weight.clone().shape_expand(context, 3)
        });
    let unroll_1 = kernel_size_1_unroll.is_some();
    let b = {
        let _lhs = {
            let _lhs = ABSOLUTE_POS::expand(context);
            let _rhs = output.clone().stride_expand(context, 0);
            burn_cube::frontend::div::expand(context, _lhs, _rhs)
        };
        let _rhs = output.clone().shape_expand(context, 0);
        burn_cube::frontend::rem::expand(context, _lhs, _rhs)
    };
    let oc = {
        let _lhs = {
            let _lhs = ABSOLUTE_POS::expand(context);
            let _rhs = output.clone().stride_expand(context, 1);
            burn_cube::frontend::div::expand(context, _lhs, _rhs)
        };
        let _rhs = output.clone().shape_expand(context, 1);
        burn_cube::frontend::rem::expand(context, _lhs, _rhs)
    };
    let oh = {
        let _lhs = {
            let _lhs = ABSOLUTE_POS::expand(context);
            let _rhs = output.clone().stride_expand(context, 2);
            burn_cube::frontend::div::expand(context, _lhs, _rhs)
        };
        let _rhs = output.clone().shape_expand(context, 2);
        burn_cube::frontend::rem::expand(context, _lhs, _rhs)
    };
    let ow = {
        let _lhs = {
            let _lhs = ABSOLUTE_POS::expand(context);
            let _rhs = output.clone().stride_expand(context, 3);
            burn_cube::frontend::div::expand(context, _lhs, _rhs)
        };
        let _rhs = output.clone().shape_expand(context, 3);
        burn_cube::frontend::rem::expand(context, _lhs, _rhs)
    };
    let g = {
        let _lhs = {
            let _lhs = weight.clone().shape_expand(context, 0);
            let _rhs = oc.clone();
            burn_cube::frontend::add::expand(context, _lhs, _rhs)
        };
        let _rhs = groups;
        burn_cube::frontend::rem::expand(context, _lhs, _rhs)
    };
    let ic_start = {
        let _lhs = in_channels.clone();
        let _rhs = g;
        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
    };
    let ic_end = {
        let _lhs = ic_start.clone();
        let _rhs = in_channels;
        burn_cube::frontend::add::expand(context, _lhs, _rhs)
    };
    let mut sum = {
        let _array = bias;
        let _index = oc.clone();
        burn_cube::frontend::index::expand(context, _array, _index)
    };
    let ih_base = {
        let _lhs = oh;
        let _rhs = conv_stride_0;
        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
    };
    let iw_base = {
        let _lhs = ow;
        let _rhs = conv_stride_1;
        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
    };
    let weight_stride_1 = weight.clone().stride_expand(context, 1);
    let weight_stride_2 = weight.clone().stride_expand(context, 2);
    let weight_stride_3 = weight.clone().stride_expand(context, 3);
    let input_stride_1 = input.clone().stride_expand(context, 1);
    let input_stride_2 = input.clone().stride_expand(context, 2);
    let input_stride_3 = input.clone().stride_expand(context, 3);
    let input_shape_2 = input.clone().shape_expand(context, 2);
    let input_shape_3 = input.clone().shape_expand(context, 3);
    let border_top = padding_0.clone();
    let border_left = padding_1.clone();
    let border_bottom = {
        let _lhs = input_shape_2;
        let _rhs = padding_0.clone();
        burn_cube::frontend::add::expand(context, _lhs, _rhs)
    };
    let border_right = {
        let _lhs = input_shape_3;
        let _rhs = padding_1.clone();
        burn_cube::frontend::add::expand(context, _lhs, _rhs)
    };
    let index_input_0 = {
        let _lhs = b;
        let _rhs = input.clone().stride_expand(context, 0);
        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
    };
    let index_weight_0 = {
        let _lhs = oc;
        let _rhs = weight.clone().stride_expand(context, 0);
        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
    };
    {
        let _start = ic_start.clone();
        let _end = ic_end;
        let _unroll = false;
        burn_cube::frontend::branch::range_expand(context, _start, _end, _unroll, |context, ic| {
            let index_input_1 = {
                let _lhs = ic.clone();
                let _rhs = input_stride_1.clone();
                burn_cube::frontend::mul::expand(context, _lhs, _rhs)
            };
            let index_weight_1 = {
                let _lhs = {
                    let _lhs = ic;
                    let _rhs = ic_start.clone();
                    burn_cube::frontend::sub::expand(context, _lhs, _rhs)
                };
                let _rhs = weight_stride_1.clone();
                burn_cube::frontend::mul::expand(context, _lhs, _rhs)
            };
            {
                let _start = 0;
                let _end = kernel_size_0.clone();
                let _unroll = unroll_0.clone();
                burn_cube::frontend::branch::range_expand(
                    context,
                    _start,
                    _end,
                    _unroll,
                    |context, kh| {
                        let _start = 0;
                        let _end = kernel_size_1.clone();
                        let _unroll = unroll_1.clone();
                        burn_cube::frontend::branch::range_expand(
                            context,
                            _start,
                            _end,
                            _unroll,
                            |context, kw| {
                                let ih = {
                                    let _lhs = {
                                        let _lhs = kh.clone();
                                        let _rhs = dilation_0.clone();
                                        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
                                    };
                                    let _rhs = ih_base.clone();
                                    burn_cube::frontend::add::expand(context, _lhs, _rhs)
                                };
                                let iw = {
                                    let _lhs = {
                                        let _lhs = kw.clone();
                                        let _rhs = dilation_1.clone();
                                        burn_cube::frontend::mul::expand(context, _lhs, _rhs)
                                    };
                                    let _rhs = iw_base.clone();
                                    burn_cube::frontend::add::expand(context, _lhs, _rhs)
                                };
                                let within_padding = {
                                    let _lhs = {
                                        let _lhs = {
                                            let _lhs = {
                                                let _lhs = ih.clone();
                                                let _rhs = border_top.clone();
                                                burn_cube::frontend::ge::expand(context, _lhs, _rhs)
                                            };
                                            let _rhs = {
                                                let _lhs = ih.clone();
                                                let _rhs = border_bottom.clone();
                                                burn_cube::frontend::lt::expand(context, _lhs, _rhs)
                                            };
                                            burn_cube::frontend::and::expand(context, _lhs, _rhs)
                                        };
                                        let _rhs = {
                                            let _lhs = iw.clone();
                                            let _rhs = border_left.clone();
                                            burn_cube::frontend::ge::expand(context, _lhs, _rhs)
                                        };
                                        burn_cube::frontend::and::expand(context, _lhs, _rhs)
                                    };
                                    let _rhs = {
                                        let _lhs = iw.clone();
                                        let _rhs = border_right.clone();
                                        burn_cube::frontend::lt::expand(context, _lhs, _rhs)
                                    };
                                    burn_cube::frontend::and::expand(context, _lhs, _rhs)
                                };
                                let _cond = within_padding;
                                burn_cube::frontend::branch::if_expand(
                                    context,
                                    None,
                                    _cond.into(),
                                    |context| {
                                        let ih_pad = {
                                            let _lhs = ih.clone();
                                            let _rhs = padding_0.clone();
                                            burn_cube::frontend::sub::expand(context, _lhs, _rhs)
                                        };
                                        let iw_pad = {
                                            let _lhs = iw.clone();
                                            let _rhs = padding_1.clone();
                                            burn_cube::frontend::sub::expand(context, _lhs, _rhs)
                                        };
                                        let index_input = {
                                            let _lhs = {
                                                let _lhs = {
                                                    let _lhs = index_input_0.clone();
                                                    let _rhs = index_input_1.clone();
                                                    burn_cube::frontend::add::expand(
                                                        context, _lhs, _rhs,
                                                    )
                                                };
                                                let _rhs = {
                                                    let _lhs = ih_pad;
                                                    let _rhs = input_stride_2.clone();
                                                    burn_cube::frontend::mul::expand(
                                                        context, _lhs, _rhs,
                                                    )
                                                };
                                                burn_cube::frontend::add::expand(
                                                    context, _lhs, _rhs,
                                                )
                                            };
                                            let _rhs = {
                                                let _lhs = iw_pad;
                                                let _rhs = input_stride_3.clone();
                                                burn_cube::frontend::mul::expand(
                                                    context, _lhs, _rhs,
                                                )
                                            };
                                            burn_cube::frontend::add::expand(context, _lhs, _rhs)
                                        };
                                        let index_weight = {
                                            let _lhs = {
                                                let _lhs = {
                                                    let _lhs = index_weight_0.clone();
                                                    let _rhs = index_weight_1.clone();
                                                    burn_cube::frontend::add::expand(
                                                        context, _lhs, _rhs,
                                                    )
                                                };
                                                let _rhs = {
                                                    let _lhs = kh.clone();
                                                    let _rhs = weight_stride_2.clone();
                                                    burn_cube::frontend::mul::expand(
                                                        context, _lhs, _rhs,
                                                    )
                                                };
                                                burn_cube::frontend::add::expand(
                                                    context, _lhs, _rhs,
                                                )
                                            };
                                            let _rhs = {
                                                let _lhs = kw.clone();
                                                let _rhs = weight_stride_3.clone();
                                                burn_cube::frontend::mul::expand(
                                                    context, _lhs, _rhs,
                                                )
                                            };
                                            burn_cube::frontend::add::expand(context, _lhs, _rhs)
                                        };
                                        {
                                            let _lhs = sum.clone();
                                            let _rhs = {
                                                let _lhs = {
                                                    let _array = input.clone();
                                                    let _index = index_input;
                                                    burn_cube::frontend::index::expand(
                                                        context, _array, _index,
                                                    )
                                                };
                                                let _rhs = {
                                                    let _array = weight.clone();
                                                    let _index = index_weight;
                                                    burn_cube::frontend::index::expand(
                                                        context, _array, _index,
                                                    )
                                                };
                                                burn_cube::frontend::mul::expand(
                                                    context, _lhs, _rhs,
                                                )
                                            };
                                            burn_cube::frontend::add_assign_op::expand(
                                                context, _lhs, _rhs,
                                            )
                                        };
                                    },
                                );
                            },
                        );
                    },
                );
            }
        });
    }
    {
        let _array = output;
        let _index = ABSOLUTE_POS::expand(context);
        let _value = sum;
        burn_cube::frontend::index_assign::expand(context, _array, _index, _value)
    };
}
pub struct Kernel<F: Float, R: Runtime> {
    settings: KernelSettings,
    kernel_size_0_unroll: Option<UInt>,
    kernel_size_1_unroll: Option<UInt>,
    _f: core::marker::PhantomData<F>,
    _r: core::marker::PhantomData<R>,
}
impl<F: Float, R: Runtime> Kernel for Kernel<F, R> {
    fn define(&self) -> KernelDefinition {
        let mut builder = KernelBuilder::default();
        let input = <Tensor<F> as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(0usize),
        );
        let weight = <Tensor<F> as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(1usize),
        );
        let bias = <Tensor<F> as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(2usize),
        );
        let conv_stride_0 = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(3usize),
        );
        let conv_stride_1 = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(4usize),
        );
        let dilation_0 = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(5usize),
        );
        let dilation_1 = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(6usize),
        );
        let padding_0 = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(7usize),
        );
        let padding_1 = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(8usize),
        );
        let groups = <UInt as LaunchArg>::compile_input(
            &mut builder,
            self.settings.vectorization_input(9usize),
        );
        let output = <Tensor<F> as LaunchArg>::compile_output(
            &mut builder,
            self.settings.vectorization_output(0usize),
        );
        kernel_expand::<F>(
            &mut builder.context,
            input,
            weight,
            bias,
            output,
            conv_stride_0,
            conv_stride_1,
            dilation_0,
            dilation_1,
            padding_0,
            padding_1,
            groups,
            self.kernel_size_0_unroll,
            self.kernel_size_1_unroll,
        );

        builder.build(self.settings.clone())
    }
    fn id(&self) -> String {
        format!(
            "{:?}-{}-{:?}-{:?}",
            core::any::TypeId::of::<Self>(),
            self.settings,
            self.kernel_size_0_unroll,
            self.kernel_size_1_unroll,
        )
    }
}
#[allow(clippy::too_many_arguments)]
pub fn kernel_launch<'a, F: Float, R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    cube_count: CubeCount,
    settings: KernelSettings,
    input: RuntimeArg<'a, Tensor<F>, R>,
    weight: RuntimeArg<'a, Tensor<F>, R>,
    bias: RuntimeArg<'a, Tensor<F>, R>,
    output: RuntimeArg<'a, Tensor<F>, R>,
    conv_stride_0: RuntimeArg<'a, UInt, R>,
    conv_stride_1: RuntimeArg<'a, UInt, R>,
    dilation_0: RuntimeArg<'a, UInt, R>,
    dilation_1: RuntimeArg<'a, UInt, R>,
    padding_0: RuntimeArg<'a, UInt, R>,
    padding_1: RuntimeArg<'a, UInt, R>,
    groups: RuntimeArg<'a, UInt, R>,
    kernel_size_0_unroll: <Comptime<Option<UInt>> as burn_cube::frontend::CubeType>::ExpandType,
    kernel_size_1_unroll: <Comptime<Option<UInt>> as burn_cube::frontend::CubeType>::ExpandType,
) -> () {
    let kernel = Kernel {
        settings,
        kernel_size_0_unroll,
        kernel_size_1_unroll,
        _f: core::marker::PhantomData::<F>,
        _r: core::marker::PhantomData::<R>,
    };
    let mut launcher = KernelLauncher::<R>::default();
    input.register(&mut launcher);
    weight.register(&mut launcher);
    bias.register(&mut launcher);
    conv_stride_0.register(&mut launcher);
    conv_stride_1.register(&mut launcher);
    dilation_0.register(&mut launcher);
    dilation_1.register(&mut launcher);
    padding_0.register(&mut launcher);
    padding_1.register(&mut launcher);
    groups.register(&mut launcher);
    output.register(&mut launcher);
    launcher.launch(cube_count, kernel, client);
}
#[cube(launch, debug)]
fn kernel<F: Float>(
    input: Tensor<F>,
    weight: Tensor<F>,
    bias: Tensor<F>,
    mut output: Tensor<F>,
    conv_stride_0: UInt,
    conv_stride_1: UInt,
    dilation_0: UInt,
    dilation_1: UInt,
    padding_0: UInt,
    padding_1: UInt,
    groups: UInt,
    kernel_size_0_unroll: Comptime<Option<UInt>>,
    kernel_size_1_unroll: Comptime<Option<UInt>>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let in_channels = weight.shape(1);

    let kernel_size_0 = Comptime::unwrap_or_else(kernel_size_0_unroll, || weight.shape(2));
    let unroll_0 = Comptime::is_some(kernel_size_0_unroll);
    let kernel_size_1 = Comptime::unwrap_or_else(kernel_size_1_unroll, || weight.shape(3));
    let unroll_1 = Comptime::is_some(kernel_size_1_unroll);

    let b = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let oc = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let oh = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let ow = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let g = (weight.shape(0) + oc) % groups;
    let ic_start = in_channels * g;
    let ic_end = ic_start + in_channels;
    let mut sum = bias[oc];

    let ih_base = oh * conv_stride_0;
    let iw_base = ow * conv_stride_1;

    let weight_stride_1 = weight.stride(1);
    let weight_stride_2 = weight.stride(2);
    let weight_stride_3 = weight.stride(3);

    let input_stride_1 = input.stride(1);
    let input_stride_2 = input.stride(2);
    let input_stride_3 = input.stride(3);
    let input_shape_2 = input.shape(2);
    let input_shape_3 = input.shape(3);

    let border_top = padding_0;
    let border_left = padding_1;
    let border_bottom = input_shape_2 + padding_0;
    let border_right = input_shape_3 + padding_1;

    let index_input_0 = b * input.stride(0);
    let index_weight_0 = oc * weight.stride(0);

    for ic in range(ic_start, ic_end, Comptime::new(false)) {
        let index_input_1 = ic * input_stride_1;
        let index_weight_1 = (ic - ic_start) * weight_stride_1;

        for kh in range(0, kernel_size_0, unroll_0) {
            for kw in range(0, kernel_size_1, unroll_1) {
                let ih = kh * dilation_0 + ih_base;
                let iw = kw * dilation_1 + iw_base;

                let within_padding = ih >= border_top
                    && ih < border_bottom
                    && iw >= border_left
                    && iw < border_right;

                if within_padding {
                    let ih_pad = ih - padding_0;
                    let iw_pad = iw - padding_1;

                    let index_input = index_input_0
                        + index_input_1
                        + ih_pad * input_stride_2
                        + iw_pad * input_stride_3;

                    let index_weight = index_weight_0
                        + index_weight_1
                        + kh * weight_stride_2
                        + kw * weight_stride_3;

                    sum += input[index_input] * weight[index_weight];
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

pub(crate) fn conv2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [out_channels, _, kernel_0, kernel_1] = weight.shape.dims;

    let out_0 = calculate_conv_output_size(
        kernel_0,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_1 = calculate_conv_output_size(
        kernel_1,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    let shape_out = Shape::new([batch_size, out_channels, out_0, out_1]);

    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    let bias = match bias {
        Some(bias) => {
            let shape = Shape::from([bias.shape.dims[0], 1, 1, 1]);
            reshape(bias, shape)
        }
        None => {
            let shape = Shape::from([output.shape.dims[0], 1, 1, 1]);
            zeros_device(input.client.clone(), input.device.clone(), shape)
        }
    };

    let num_elems_output = output.shape.num_elements();
    let workgroup = calculate_cube_count_elemwise(num_elems_output, SUBCUBE_DIM_APPROX);
    let settings = KernelSettings::default()
        .vectorize_input(0, 1)
        .vectorize_output(0, 1);

    kernel_launch::<E::CubeElement, R>(
        input.client,
        workgroup,
        settings,
        TensorHandle::new(&input.handle, &input.strides, &input.shape.dims),
        TensorHandle::new(&weight.handle, &weight.strides, &weight.shape.dims),
        TensorHandle::new(&bias.handle, &bias.strides, &bias.shape.dims),
        TensorHandle::new(&output.handle, &output.strides, &output.shape.dims),
        options.stride[0] as u32,
        options.stride[1] as u32,
        options.dilation[0] as u32,
        options.dilation[1] as u32,
        options.padding[0] as u32,
        options.padding[1] as u32,
        options.groups as u32,
        Some(kernel_0.into()),
        Some(kernel_1.into()),
    );

    output
}
