use std::marker::PhantomData;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::{gpu, ComputeShader, Elem, IntKind, Scope, Variable, Visibility},
    kernel::{self, GpuComputeShaderPhase},
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::{ops::ConvTransposeOptions, Element, Shape};

#[derive(new)]
struct Conv2dTransposeEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

#[derive(new)]
struct Conv2dTransposeComputeShader<E> {
    input: Variable,
    weight: Variable,
    bias: Variable,
    output: Variable,
    _elem: PhantomData<E>,
}

impl<E: JitElement> Conv2dTransposeComputeShader<E> {
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
        gpu!(scope, input_stride_0 = stride(input, 0u32));
        gpu!(scope, input_stride_1 = stride(input, 1u32));
        gpu!(scope, input_stride_2 = stride(input, 2u32));
        gpu!(scope, input_stride_3 = stride(input, 3u32));
        gpu!(scope, input_shape_0 = shape(input, 0u32));
        gpu!(scope, input_shape_1 = shape(input, 1u32));
        gpu!(scope, input_shape_2 = shape(input, 2u32));
        gpu!(scope, input_shape_3 = shape(input, 3u32));

        let output_stride_0 = scope.create_local(Elem::UInt);
        let output_stride_1 = scope.create_local(Elem::UInt);
        let output_stride_2 = scope.create_local(Elem::UInt);
        let output_stride_3 = scope.create_local(Elem::UInt);
        let output_shape_0 = scope.create_local(Elem::UInt);
        let output_shape_1 = scope.create_local(Elem::UInt);
        let output_shape_2 = scope.create_local(Elem::UInt);
        let output_shape_3 = scope.create_local(Elem::UInt);
        gpu!(scope, output_stride_0 = stride(output, 0u32));
        gpu!(scope, output_stride_1 = stride(output, 1u32));
        gpu!(scope, output_stride_2 = stride(output, 2u32));
        gpu!(scope, output_stride_3 = stride(output, 3u32));
        gpu!(scope, output_shape_0 = shape(output, 0u32));
        gpu!(scope, output_shape_1 = shape(output, 1u32));
        gpu!(scope, output_shape_2 = shape(output, 2u32));
        gpu!(scope, output_shape_3 = shape(output, 3u32));

        let weight_stride_0 = scope.create_local(Elem::UInt);
        let weight_stride_1 = scope.create_local(Elem::UInt);
        let weight_stride_2 = scope.create_local(Elem::UInt);
        let weight_stride_3 = scope.create_local(Elem::UInt);
        let in_channels = scope.create_local(Elem::UInt);
        let weight_shape_1 = scope.create_local(Elem::UInt);
        let kernel_size_0 = scope.create_local(Elem::UInt);
        let kernel_size_1 = scope.create_local(Elem::UInt);
        gpu!(scope, weight_stride_0 = stride(weight, 0u32));
        gpu!(scope, weight_stride_1 = stride(weight, 1u32));
        gpu!(scope, weight_stride_2 = stride(weight, 2u32));
        gpu!(scope, weight_stride_3 = stride(weight, 3u32));
        gpu!(scope, in_channels = shape(weight, 0u32));
        gpu!(scope, weight_shape_1 = shape(weight, 1u32));
        gpu!(scope, kernel_size_0 = shape(weight, 2u32));
        gpu!(scope, kernel_size_1 = shape(weight, 3u32));

        let conv_stride_0 = Variable::GlobalScalar(0, Elem::UInt);
        let conv_stride_1 = Variable::GlobalScalar(1, Elem::UInt);
        let dilation_0 = Variable::GlobalScalar(2, Elem::UInt);
        let dilation_1 = Variable::GlobalScalar(3, Elem::UInt);
        let padding_0 = Variable::GlobalScalar(4, Elem::UInt);
        let padding_1 = Variable::GlobalScalar(5, Elem::UInt);
        let groups = Variable::GlobalScalar(6, Elem::UInt);

        let stride_0_i = scope.create_local(Elem::Int(IntKind::I32));
        let stride_1_i = scope.create_local(Elem::Int(IntKind::I32));
        gpu!(scope, stride_0_i = cast(conv_stride_0));
        gpu!(scope, stride_1_i = cast(conv_stride_1));

        let oc_out = scope.create_local(Elem::UInt);
        let oc = scope.create_local(Elem::UInt);

        let b = scope.create_local(Elem::UInt);
        let oh = scope.create_local(Elem::UInt);
        let ow = scope.create_local(Elem::UInt);
        let k = scope.create_local(Elem::UInt);
        let g = scope.create_local(Elem::UInt);

        let ic_start = scope.create_local(Elem::UInt);
        let ic_end = scope.create_local(Elem::UInt);
        let ic_tmp = scope.create_local(Elem::UInt);

        gpu!(scope, b = id / output_stride_0);
        gpu!(scope, b = b % output_shape_0);

        gpu!(scope, oc_out = id / output_stride_1);
        gpu!(scope, oc_out = oc_out % output_shape_1);

        gpu!(scope, oh = id / output_stride_2);
        gpu!(scope, oh = oh % output_shape_2);

        gpu!(scope, ow = id / output_stride_3);
        gpu!(scope, ow = ow % output_shape_3);

        gpu!(scope, k = oc_out / weight_shape_1);
        gpu!(scope, g = k % groups);
        gpu!(scope, oc = weight_shape_1 * g);
        gpu!(scope, oc = oc_out - oc);

        gpu!(scope, ic_tmp = in_channels / groups);
        gpu!(scope, ic_start = g * ic_tmp);
        gpu!(scope, ic_end = ic_start + ic_tmp);

        let tmp_u = scope.create_local(Elem::UInt);
        let tmp_i = scope.create_local(Elem::Int(IntKind::I32));
        let zero_i = scope.zero(Elem::Int(IntKind::I32));
        let one_i = scope.create_with_value(1, Elem::Int(IntKind::I32));

        let kms_u = scope.create_local(Elem::UInt);
        let kms_0 = scope.create_local(Elem::Int(IntKind::I32));
        let kms_1 = scope.create_local(Elem::Int(IntKind::I32));
        let ih_start_tmp = scope.create_local(Elem::Int(IntKind::I32));
        let iw_start_tmp = scope.create_local(Elem::Int(IntKind::I32));
        let ih_start = scope.create_local(Elem::UInt);
        let iw_start = scope.create_local(Elem::UInt);
        let ih_end = scope.create_local(Elem::UInt);
        let iw_end = scope.create_local(Elem::UInt);

        gpu!(scope, kms_u = kernel_size_0 * dilation_0);
        gpu!(scope, kms_0 = cast(kms_u));
        gpu!(scope, kms_0 = kms_0 - stride_0_i);
        gpu!(scope, kms_u = kernel_size_1 * dilation_1);
        gpu!(scope, kms_1 = cast(kms_u));
        gpu!(scope, kms_1 = kms_1 - stride_1_i);

        gpu!(scope, tmp_u = oh + padding_0);
        gpu!(scope, tmp_i = cast(tmp_u));
        gpu!(scope, ih_start_tmp = tmp_i - kms_0);
        gpu!(scope, ih_start_tmp = ih_start_tmp / stride_0_i);
        gpu!(scope, tmp_u = ow + padding_1);
        gpu!(scope, tmp_i = cast(tmp_u));
        gpu!(scope, iw_start_tmp = tmp_i - kms_1);
        gpu!(scope, iw_start_tmp = iw_start_tmp / stride_1_i);

        gpu!(scope, tmp_i = max(ih_start_tmp, zero_i));
        gpu!(scope, ih_start = cast(tmp_i));
        gpu!(scope, tmp_i = kms_0 + ih_start_tmp);
        gpu!(scope, tmp_i += one_i);
        gpu!(scope, tmp_i = max(tmp_i, zero_i));
        gpu!(scope, tmp_u = cast(tmp_i));
        gpu!(scope, ih_end = min(tmp_u, input_shape_2));

        gpu!(scope, tmp_i = max(iw_start_tmp, zero_i));
        gpu!(scope, iw_start = cast(tmp_i));
        gpu!(scope, tmp_i = kms_1 + iw_start_tmp);
        gpu!(scope, tmp_i += one_i);
        gpu!(scope, tmp_i = max(tmp_i, zero_i));
        gpu!(scope, tmp_u = cast(tmp_i));
        gpu!(scope, iw_end = min(tmp_u, input_shape_3));

        let index_input = scope.create_local(Elem::UInt);
        let index_weight = scope.create_local(Elem::UInt);

        let index_input_b = scope.create_local(Elem::UInt);
        let index_input_ic = scope.create_local(Elem::UInt);
        let index_input_ih = scope.create_local(Elem::UInt);
        let index_input_iw = scope.create_local(Elem::UInt);
        let index_weight_ic = scope.create_local(Elem::UInt);
        let index_weight_oc = scope.create_local(Elem::UInt);
        let index_weight_kh = scope.create_local(Elem::UInt);
        let index_weight_kw = scope.create_local(Elem::UInt);

        gpu!(scope, index_input_b = b * input_stride_0);
        gpu!(scope, index_weight_oc = oc * weight_stride_1);

        let prod = scope.create_local(output.item());
        let prod_tmp = scope.create_local(output.item());
        let sum = scope.create_local(output.item());
        gpu!(scope, sum = bias[oc_out]);

        let kh = scope.create_local(Elem::UInt);
        let kw = scope.create_local(Elem::UInt);
        let numerator_h_base = scope.create_local(Elem::UInt);
        let numerator_h = scope.create_local(Elem::UInt);
        let numerator_w_base = scope.create_local(Elem::UInt);
        let numerator_w = scope.create_local(Elem::UInt);
        let numerator_tmp = scope.create_local(Elem::UInt);
        let numerator_mod = scope.create_local(Elem::UInt);
        let zero = scope.zero(Elem::UInt);
        let divisible = scope.create_local(Elem::Bool);
        let not_neg = scope.create_local(Elem::Bool);
        let cond = scope.create_local(Elem::Bool);

        gpu!(scope, numerator_h_base = oh + padding_0);
        gpu!(scope, numerator_w_base = ow + padding_1);

        gpu!(
            scope,
            range(ic_start, ic_end).for_each(|ic, scope| {
                gpu!(scope, index_input_ic = ic * input_stride_1);
                gpu!(scope, index_weight_ic = ic * weight_stride_0);

                gpu!(
                    scope,
                    range(ih_start, ih_end).for_each(|ih, scope| {
                        gpu!(scope, numerator_tmp = ih * conv_stride_0);
                        gpu!(scope, not_neg = numerator_h_base >= numerator_tmp);
                        gpu!(scope, numerator_h = numerator_h_base - numerator_tmp);

                        gpu!(scope, numerator_mod = numerator_h % dilation_0);
                        gpu!(scope, divisible = numerator_mod == zero);
                        gpu!(scope, cond = not_neg && divisible);

                        gpu!(scope, if(cond).then(|scope|{
                            gpu!(scope, kh = numerator_h / dilation_0);
                            gpu!(scope, index_input_ih = ih * input_stride_2);
                            gpu!(scope, index_weight_kh = kh * weight_stride_2);

                            gpu!(
                                scope,
                                range(iw_start, iw_end).for_each(|iw, scope| {
                                    gpu!(scope, numerator_tmp = iw * conv_stride_1);
                                    gpu!(scope, not_neg = numerator_w_base >= numerator_tmp);
                                    gpu!(scope, numerator_w = numerator_w_base - numerator_tmp);

                                    gpu!(scope, numerator_mod = numerator_w % dilation_1);
                                    gpu!(scope, divisible = numerator_mod == zero);
                                    gpu!(scope, cond = not_neg && divisible);

                                    gpu!(scope, if(cond).then(|scope|{
                                        gpu!(scope, kw = numerator_w / dilation_1);
                                        gpu!(scope, index_input_iw = iw * input_stride_3);
                                        gpu!(scope, index_weight_kw = kw * weight_stride_3);

                                        gpu!(scope, index_input = index_input_b);
                                        gpu!(scope, index_input += index_input_ic);
                                        gpu!(scope, index_input += index_input_ih);
                                        gpu!(scope, index_input += index_input_iw);

                                        gpu!(scope, index_weight = index_weight_ic);
                                        gpu!(scope, index_weight += index_weight_oc);
                                        gpu!(scope, index_weight += index_weight_kh);
                                        gpu!(scope, index_weight += index_weight_kw);

                                        gpu!(scope, prod = input[index_input]);
                                        gpu!(scope, prod_tmp = weight[index_weight]);
                                        gpu!(scope, prod *= prod_tmp);
                                        gpu!(scope, sum += prod);
                                    }));
                                })
                            );

                        }));
                    })
                );
            })
        );

        gpu!(scope, output[id] = sum);
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for Conv2dTransposeEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let weight = Variable::GlobalInputArray(1, item);
        let bias = Variable::GlobalInputArray(2, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        Conv2dTransposeComputeShader {
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
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub(crate) fn conv_transpose2d<R: Runtime, E: JitElement + Element>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = kernel::into_contiguous(input);
    let weight = kernel::into_contiguous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [_, out_channels, kernel_0, kernel_1] = weight.shape.dims;

    let out_0 = (in_height - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_width - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;

    let shape_out = Shape::new([batch_size, out_channels * options.groups, out_0, out_1]);

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

    let kernel = Conv2dTransposeEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client.clone())
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
