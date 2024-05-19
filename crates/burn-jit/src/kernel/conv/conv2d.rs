use burn_cube::{
    cpa,
    dialect::{ComputeShader, Elem, Scope, Variable, Visibility},
    Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
    OutputInfo, WorkgroupLaunch,
};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use std::marker::PhantomData;

use crate::{
    element::JitElement,
    kernel::{into_contiguous, GpuComputeShaderPhase},
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    JitRuntime,
};

#[derive(new)]
struct Conv2dEagerKernel<R: JitRuntime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct Conv2dComputeShader<E: JitElement> {
    input: Variable,
    weight: Variable,
    bias: Variable,
    output: Variable,
    _elem: PhantomData<E>,
}

impl<E: JitElement> Conv2dComputeShader<E> {
    fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let weight = self.weight;
        let bias = self.bias;
        let output = self.output;
        let id = Variable::Id;

        let input_stride_0 = scope.create_local(Elem::UInt);
        let input_stride_1 = scope.create_local(Elem::UInt);
        let input_stride_2 = scope.create_local(Elem::UInt);
        let input_stride_3 = scope.create_local(Elem::UInt);
        let input_shape_0 = scope.create_local(Elem::UInt);
        let input_shape_1 = scope.create_local(Elem::UInt);
        let input_shape_2 = scope.create_local(Elem::UInt);
        let input_shape_3 = scope.create_local(Elem::UInt);
        cpa!(scope, input_stride_0 = stride(input, 0u32));
        cpa!(scope, input_stride_1 = stride(input, 1u32));
        cpa!(scope, input_stride_2 = stride(input, 2u32));
        cpa!(scope, input_stride_3 = stride(input, 3u32));
        cpa!(scope, input_shape_0 = shape(input, 0u32));
        cpa!(scope, input_shape_1 = shape(input, 1u32));
        cpa!(scope, input_shape_2 = shape(input, 2u32));
        cpa!(scope, input_shape_3 = shape(input, 3u32));

        let output_stride_0 = scope.create_local(Elem::UInt);
        let output_stride_1 = scope.create_local(Elem::UInt);
        let output_stride_2 = scope.create_local(Elem::UInt);
        let output_stride_3 = scope.create_local(Elem::UInt);
        let output_shape_0 = scope.create_local(Elem::UInt);
        let output_shape_1 = scope.create_local(Elem::UInt);
        let output_shape_2 = scope.create_local(Elem::UInt);
        let output_shape_3 = scope.create_local(Elem::UInt);
        cpa!(scope, output_stride_0 = stride(output, 0u32));
        cpa!(scope, output_stride_1 = stride(output, 1u32));
        cpa!(scope, output_stride_2 = stride(output, 2u32));
        cpa!(scope, output_stride_3 = stride(output, 3u32));
        cpa!(scope, output_shape_0 = shape(output, 0u32));
        cpa!(scope, output_shape_1 = shape(output, 1u32));
        cpa!(scope, output_shape_2 = shape(output, 2u32));
        cpa!(scope, output_shape_3 = shape(output, 3u32));

        let weight_stride_0 = scope.create_local(Elem::UInt);
        let weight_stride_1 = scope.create_local(Elem::UInt);
        let weight_stride_2 = scope.create_local(Elem::UInt);
        let weight_stride_3 = scope.create_local(Elem::UInt);
        let weight_shape_0 = scope.create_local(Elem::UInt);
        let in_channels = scope.create_local(Elem::UInt);
        let kernel_size_0 = scope.create_local(Elem::UInt);
        let kernel_size_1 = scope.create_local(Elem::UInt);
        cpa!(scope, weight_stride_0 = stride(weight, 0u32));
        cpa!(scope, weight_stride_1 = stride(weight, 1u32));
        cpa!(scope, weight_stride_2 = stride(weight, 2u32));
        cpa!(scope, weight_stride_3 = stride(weight, 3u32));
        cpa!(scope, weight_shape_0 = shape(weight, 0u32));
        cpa!(scope, in_channels = shape(weight, 1u32));
        cpa!(scope, kernel_size_0 = shape(weight, 2u32));
        cpa!(scope, kernel_size_1 = shape(weight, 3u32));

        let conv_stride_0 = Variable::GlobalScalar(0, Elem::UInt);
        let conv_stride_1 = Variable::GlobalScalar(1, Elem::UInt);
        let dilation_0 = Variable::GlobalScalar(2, Elem::UInt);
        let dilation_1 = Variable::GlobalScalar(3, Elem::UInt);
        let padding_0 = Variable::GlobalScalar(4, Elem::UInt);
        let padding_1 = Variable::GlobalScalar(5, Elem::UInt);
        let groups = Variable::GlobalScalar(6, Elem::UInt);

        let b = scope.create_local(Elem::UInt);
        let oc = scope.create_local(Elem::UInt);
        let oh = scope.create_local(Elem::UInt);
        let ow = scope.create_local(Elem::UInt);
        let g = scope.create_local(Elem::UInt);

        let ic_start = scope.create_local(Elem::UInt);
        let ic_end = scope.create_local(Elem::UInt);

        cpa!(scope, b = id / output_stride_0);
        cpa!(scope, b = b % output_shape_0);

        cpa!(scope, oc = id / output_stride_1);
        cpa!(scope, oc = oc % output_shape_1);

        cpa!(scope, oh = id / output_stride_2);
        cpa!(scope, oh = oh % output_shape_2);

        cpa!(scope, ow = id / output_stride_3);
        cpa!(scope, ow = ow % output_shape_3);

        cpa!(scope, g = weight_shape_0 + oc);
        cpa!(scope, g = g % groups);

        cpa!(scope, ic_start = in_channels * g);
        cpa!(scope, ic_end = ic_start + in_channels);

        let sum = scope.create_local(output.item());
        cpa!(scope, sum = bias[oc]);

        let ih_base = scope.create_local(Elem::UInt);
        let iw_base = scope.create_local(Elem::UInt);
        let ih = scope.create_local(Elem::UInt);
        let iw = scope.create_local(Elem::UInt);

        let padding = scope.create_local(Elem::Bool);
        let padding_accumulator = scope.create_local(Elem::Bool);
        let border_top = scope.create_local(Elem::UInt);
        let border_bottom = scope.create_local(Elem::UInt);
        let border_left = scope.create_local(Elem::UInt);
        let border_right = scope.create_local(Elem::UInt);

        let ih_pad = scope.create_local(Elem::UInt);
        let iw_pad = scope.create_local(Elem::UInt);

        let index_input = scope.create_local(Elem::UInt);
        let index_input_0 = scope.create_local(Elem::UInt);
        let index_input_1 = scope.create_local(Elem::UInt);
        let index_input_2 = scope.create_local(Elem::UInt);
        let index_input_3 = scope.create_local(Elem::UInt);

        let index_weight = scope.create_local(Elem::UInt);
        let index_weight_0 = scope.create_local(Elem::UInt);
        let index_weight_1 = scope.create_local(Elem::UInt);
        let index_weight_2 = scope.create_local(Elem::UInt);
        let index_weight_3 = scope.create_local(Elem::UInt);

        let input_value = scope.create_local(input.item());
        let weight_value = scope.create_local(weight.item());
        let value_product = scope.create_local(input.item());

        cpa!(scope, ih_base = oh * conv_stride_0);
        cpa!(scope, iw_base = ow * conv_stride_1);

        cpa!(scope, border_top = padding_0);
        cpa!(scope, border_left = padding_1);
        cpa!(scope, border_bottom = input_shape_2 + padding_0);
        cpa!(scope, border_right = input_shape_3 + padding_1);

        cpa!(scope, index_input_0 = b * input_stride_0);
        cpa!(scope, index_weight_0 = oc * weight_stride_0);

        cpa!(
            scope,
            range(ic_start, ic_end).for_each(|ic, scope| {
                cpa!(scope, index_input_1 = ic * input_stride_1);
                cpa!(scope, index_weight_1 = ic - ic_start);
                cpa!(scope, index_weight_1 *= weight_stride_1);

                cpa!(
                    scope,
                    range(0u32, kernel_size_0).for_each(|kh, scope| {
                        cpa!(
                            scope,
                            range(0u32, kernel_size_1).for_each(|kw, scope| {
                                cpa!(scope, ih = kh * dilation_0);
                                cpa!(scope, ih += ih_base);
                                cpa!(scope, iw = kw * dilation_1);
                                cpa!(scope, iw += iw_base);

                                cpa!(scope, padding_accumulator = ih >= border_top);
                                cpa!(scope, padding = ih < border_bottom);
                                cpa!(scope, padding_accumulator = padding_accumulator && padding);
                                cpa!(scope, padding = iw >= border_left);
                                cpa!(scope, padding_accumulator = padding_accumulator && padding);
                                cpa!(scope, padding = iw < border_right);
                                cpa!(scope, padding_accumulator = padding_accumulator && padding);

                                cpa!(scope, if(padding_accumulator).then(|scope|{
                                    cpa!(scope, ih_pad = ih - padding_0);
                                    cpa!(scope, iw_pad = iw - padding_1);

                                    cpa!(scope, index_input_2 = ih_pad * input_stride_2);
                                    cpa!(scope, index_input_3 = iw_pad * input_stride_3);

                                    cpa!(scope, index_input = index_input_0);
                                    cpa!(scope, index_input += index_input_1);
                                    cpa!(scope, index_input += index_input_2);
                                    cpa!(scope, index_input += index_input_3);

                                    cpa!(scope, index_weight_2 = kh * weight_stride_2);
                                    cpa!(scope, index_weight_3 = kw * weight_stride_3);

                                    cpa!(scope, index_weight = index_weight_0);
                                    cpa!(scope, index_weight += index_weight_1);
                                    cpa!(scope, index_weight += index_weight_2);
                                    cpa!(scope, index_weight += index_weight_3);

                                    cpa!(scope, input_value = input[index_input]);
                                    cpa!(scope, weight_value = weight[index_weight]);
                                    cpa!(scope, value_product = input_value * weight_value);
                                    cpa!(scope, sum += value_product);
                                }));
                            })
                        );
                    })
                );
            })
        );

        cpa!(scope, output[id] = sum);
    }
}

impl<R: JitRuntime, E: JitElement> GpuComputeShaderPhase for Conv2dEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let weight = Variable::GlobalInputArray(1, item);
        let bias = Variable::GlobalInputArray(2, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        Conv2dComputeShader {
            input,
            weight,
            bias,
            output,
            _elem: PhantomData::<E>,
        }
        .expand(&mut scope);

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

pub(crate) fn conv2d<R: JitRuntime, E: JitElement>(
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
