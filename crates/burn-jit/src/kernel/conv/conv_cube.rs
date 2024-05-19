use std::marker::PhantomData;

use burn_cube::{
    branch::range,
    dialect::{ComputeShader, Elem, Visibility},
    *,
};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

use crate::{
    fusion::kernel,
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[cube]
fn convolution<F: Float>(
    input: Array<F>,
    weight: Array<F>,
    bias: Array<F>,
    mut output: Array<F>,
    conv_stride_0: UInt,
    conv_stride_1: UInt,
    dilation_0: UInt,
    dilation_1: UInt,
    padding_0: UInt,
    padding_1: UInt,
    groups: UInt,
) {
    let output_stride_0 = stride::<F>(output, 0u32);
    let output_stride_1 = stride::<F>(output, 1u32);
    let output_stride_2 = stride::<F>(output, 2u32);
    let output_stride_3 = stride::<F>(output, 3u32);
    let output_shape_0 = shape::<F>(output, 0u32);
    let output_shape_1 = shape::<F>(output, 1u32);
    let output_shape_2 = shape::<F>(output, 2u32);
    let output_shape_3 = shape::<F>(output, 3u32);

    let weight_shape_0 = shape::<F>(weight, 0u32);
    let in_channels = shape::<F>(weight, 1u32);
    let kernel_size_0 = shape::<F>(weight, 2u32);
    let kernel_size_1 = shape::<F>(weight, 3u32);

    let b = AbsoluteIndex::get() / output_stride_0 % output_shape_0;
    let oc = AbsoluteIndex::get() / output_stride_1 % output_shape_1;
    let oh = AbsoluteIndex::get() / output_stride_2 % output_shape_2;
    let ow = AbsoluteIndex::get() / output_stride_3 % output_shape_3;
    let g = (weight_shape_0 + oc) % groups;
    let ic_start = in_channels * g;
    let ic_end = ic_start + in_channels;
    let mut sum = bias[oc];

    let ih_base = oh * conv_stride_0;
    let iw_base = ow * conv_stride_1;

    let input_stride_0 = stride::<F>(input, 0u32);
    let input_stride_1 = stride::<F>(input, 1u32);
    let input_stride_2 = stride::<F>(input, 2u32);
    let input_stride_3 = stride::<F>(input, 3u32);
    let input_shape_2 = shape::<F>(input, 2u32);
    let input_shape_3 = shape::<F>(input, 3u32);

    let border_top = padding_0;
    let border_left = padding_1;
    let border_bottom = input_shape_2 + padding_0;
    let border_right = input_shape_3 + padding_1;

    let weight_stride_0 = stride::<F>(weight, 0u32);
    let weight_stride_1 = stride::<F>(weight, 1u32);
    let weight_stride_2 = stride::<F>(weight, 2u32);
    let weight_stride_3 = stride::<F>(weight, 3u32);

    let index_input_0 = b * input_stride_0;
    let index_weight_0 = oc * weight_stride_0;

    for ic in range(ic_start, ic_end, false) {
        let index_input_1 = ic * input_stride_1;
        let index_weight_1 = (ic - ic_start) * weight_stride_1;

        for kh in range(0u32, kernel_size_0, false) {
            for kw in range(0u32, kernel_size_1, false) {
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

    output[AbsoluteIndex::get()] = sum;
}

#[derive(new)]
struct Conv2dEagerKernel<R: JitRuntime, E: FloatElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct Conv2dComputeShader<E: FloatElement> {
    input: ExpandElement,
    weight: ExpandElement,
    bias: ExpandElement,
    output: ExpandElement,
    conv_stride_0: ExpandElement,
    conv_stride_1: ExpandElement,
    dilation_0: ExpandElement,
    dilation_1: ExpandElement,
    padding_0: ExpandElement,
    padding_1: ExpandElement,
    groups: ExpandElement,
    _elem: PhantomData<E>,
}

impl<R: JitRuntime, E: FloatElement> GpuComputeShaderPhase for Conv2dEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut context = CubeContext::root();
        let item = E::cube_elem().into();

        let input = context.input(0, item);
        let weight = context.input(1, item);
        let bias = context.input(2, item);
        let output = context.output(0, item);
        let conv_stride_0 = context.scalar(0, Elem::UInt);
        let conv_stride_1 = context.scalar(1, Elem::UInt);
        let dilation_0 = context.scalar(2, Elem::UInt);
        let dilation_1 = context.scalar(3, Elem::UInt);
        let padding_0 = context.scalar(4, Elem::UInt);
        let padding_1 = context.scalar(5, Elem::UInt);
        let groups = context.scalar(6, Elem::UInt);

        Conv2dComputeShader {
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
            _elem: PhantomData::<E>,
        }
        .expand(&mut context);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let weight = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let bias = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let scalars = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 7,
        };

        let output = OutputInfo::Array { item };

        let scope = context.into_scope();
        let info = CompilationInfo {
            inputs: vec![input, weight, bias, scalars],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>(),)
    }
}

impl<E: FloatElement> Conv2dComputeShader<E> {
    fn expand(self, context: &mut CubeContext) {
        convolution_expand::<E::CubeElement>(
            context,
            self.input,
            self.weight,
            self.bias,
            self.output,
            self.conv_stride_0,
            self.conv_stride_1,
            self.dilation_0,
            self.dilation_1,
            self.padding_0,
            self.padding_1,
            self.groups,
        )
    }
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

    let kernel = Conv2dEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client)
        .inputs(&[
            EagerHandle::<R>::new(&input.handle, &input.strides, &input.shape.dims),
            EagerHandle::new(&weight.handle, &weight.strides, &weight.shape.dims),
            EagerHandle::new(&bias.handle, &bias.strides, &bias.shape.dims),
        ])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .with_scalars(&[
            options.stride[0] as u32,
            options.stride[1] as u32,
            options.dilation[0] as u32,
            options.dilation[1] as u32,
            options.padding[0] as u32,
            options.padding[1] as u32,
            options.groups as u32,
        ])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
