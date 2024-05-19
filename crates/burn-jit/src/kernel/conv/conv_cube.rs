use std::marker::PhantomData;

use burn_cube::{
    cube,
    dialect::{ComputeShader, Elem, Scope, Variable, Visibility},
    Array, Cast, Compilation, CompilationInfo, CompilationSettings, CubeContext, EagerHandle,
    Execution, ExpandElement, Float, GpuComputeShaderPhase, InputInfo, OutputInfo, UInt,
    WorkgroupLaunch, F32, I32,
};
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
    let x = input[UInt::new(0u32)];
    let w = weight[UInt::new(0u32)];
    let b = bias[UInt::new(0u32)];
    output[UInt::new(0u32)] = F::cast_from(padding_0);
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
