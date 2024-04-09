use std::marker::PhantomData;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, ComputeShader, Elem, Scope, Variable, Visibility},
    kernel::GpuComputeShaderPhase,
    tensor::JitTensor,
    JitElement, Runtime,
};

#[derive(new)]
struct InterpolateBilinearEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct InterpolateBilinearShader {
    input: Variable,
    output: Variable,
}

impl InterpolateBilinearShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::Id;

        let input_stride_0 = scope.create_local(Elem::UInt);
        let input_stride_1 = scope.create_local(Elem::UInt);
        let input_stride_2 = scope.create_local(Elem::UInt);
        let input_stride_3 = scope.create_local(Elem::UInt);

        let input_shape_2 = scope.create_local(Elem::UInt);
        let input_shape_3 = scope.create_local(Elem::UInt);

        let output_stride_0 = scope.create_local(Elem::UInt);
        let output_stride_1 = scope.create_local(Elem::UInt);
        let output_stride_2 = scope.create_local(Elem::UInt);
        let output_stride_3 = scope.create_local(Elem::UInt);

        let output_shape_0 = scope.create_local(Elem::UInt);
        let output_shape_1 = scope.create_local(Elem::UInt);
        let output_shape_2 = scope.create_local(Elem::UInt);
        let output_shape_3 = scope.create_local(Elem::UInt);

        gpu!(scope, input_stride_0 = stride(input, 0u32));
        gpu!(scope, input_stride_1 = stride(input, 1u32));
        gpu!(scope, input_stride_2 = stride(input, 2u32));
        gpu!(scope, input_stride_3 = stride(input, 3u32));

        gpu!(scope, input_shape_2 = shape(input, 2u32));
        gpu!(scope, input_shape_3 = shape(input, 3u32));

        gpu!(scope, output_stride_0 = stride(output, 0u32));
        gpu!(scope, output_stride_1 = stride(output, 1u32));
        gpu!(scope, output_stride_2 = stride(output, 2u32));
        gpu!(scope, output_stride_3 = stride(output, 3u32));

        gpu!(scope, output_shape_0 = shape(output, 0u32));
        gpu!(scope, output_shape_1 = shape(output, 1u32));
        gpu!(scope, output_shape_2 = shape(output, 2u32));
        gpu!(scope, output_shape_3 = shape(output, 3u32));

        let b = scope.create_local(Elem::UInt);
        let c = scope.create_local(Elem::UInt);
        let h = scope.create_local(Elem::UInt);
        let w = scope.create_local(Elem::UInt);

        gpu!(scope, b = id / output_stride_0);
        gpu!(scope, b = b % output_shape_0);

        gpu!(scope, c = id / output_stride_1);
        gpu!(scope, c = c % output_shape_1);

        gpu!(scope, h = id / output_stride_2);
        gpu!(scope, h = h % output_shape_2);

        gpu!(scope, w = id / output_stride_3);
        gpu!(scope, w = w % output_shape_3);

        let factor_float = scope.create_local(input.item());
        let numerator_float = scope.create_local(input.item());
        let numerator_int = scope.create_local(Elem::UInt);
        let denominator_float = scope.create_local(input.item());
        let denominator_int = scope.create_local(Elem::UInt);

        let frac = scope.create_local(input.item());
        let v0 = scope.create_local(input.item());
        let v1 = scope.create_local(input.item());
        let one = scope.create_with_value(1f32, input.item());

        let y0 = scope.create_local(Elem::UInt);
        let y1 = scope.create_local(Elem::UInt);
        let yw = scope.create_local(input.item());
        let yw_ = scope.create_local(input.item());

        let x0 = scope.create_local(Elem::UInt);
        let x1 = scope.create_local(Elem::UInt);
        let xw = scope.create_local(input.item());
        let xw_ = scope.create_local(input.item());

        gpu!(scope, numerator_int = input_shape_2 - 1u32);
        gpu!(scope, denominator_int = output_shape_2 - 1u32);
        gpu!(scope, factor_float = cast(h));
        gpu!(scope, numerator_float = cast(numerator_int));
        gpu!(scope, denominator_float = cast(denominator_int));
        gpu!(scope, frac = factor_float * numerator_float);
        gpu!(scope, frac = frac / denominator_float);
        gpu!(scope, v0 = floor(frac));
        gpu!(scope, v1 = ceil(frac));
        gpu!(scope, yw = frac - v0);
        gpu!(scope, yw_ = one - yw);
        gpu!(scope, y0 = cast(v0));
        gpu!(scope, y1 = cast(v1));

        gpu!(scope, numerator_int = input_shape_3 - 1u32);
        gpu!(scope, denominator_int = output_shape_3 - 1u32);
        gpu!(scope, factor_float = cast(w));
        gpu!(scope, numerator_float = cast(numerator_int));
        gpu!(scope, denominator_float = cast(denominator_int));
        gpu!(scope, frac = factor_float * numerator_float);
        gpu!(scope, frac = frac / denominator_float);
        gpu!(scope, v0 = floor(frac));
        gpu!(scope, v1 = ceil(frac));
        gpu!(scope, xw = frac - v0);
        gpu!(scope, xw_ = one - xw);
        gpu!(scope, x0 = cast(v0));
        gpu!(scope, x1 = cast(v1));

        let index_base = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        let index = scope.create_local(Elem::UInt);
        let y0_stride = scope.create_local(Elem::UInt);
        let y1_stride = scope.create_local(Elem::UInt);
        let x0_stride = scope.create_local(Elem::UInt);
        let x1_stride = scope.create_local(Elem::UInt);
        let p_a = scope.create_local(input.item());
        let p_b = scope.create_local(input.item());
        let p_c = scope.create_local(input.item());
        let p_d = scope.create_local(input.item());

        gpu!(scope, index_base = b * input_stride_0);
        gpu!(scope, index_tmp = c * input_stride_1);
        gpu!(scope, index_base += index_tmp);
        gpu!(scope, y0_stride = y0 * input_stride_2);
        gpu!(scope, y1_stride = y1 * input_stride_2);
        gpu!(scope, x0_stride = x0 * input_stride_3);
        gpu!(scope, x1_stride = x1 * input_stride_3);

        gpu!(scope, index = index_base);
        gpu!(scope, index += y0_stride);
        gpu!(scope, index += x0_stride);
        gpu!(scope, p_a = input[index]);
        gpu!(scope, p_a *= xw_);
        gpu!(scope, p_a *= yw_);

        gpu!(scope, index = index_base);
        gpu!(scope, index += y0_stride);
        gpu!(scope, index += x1_stride);
        gpu!(scope, p_b = input[index]);
        gpu!(scope, p_b *= xw);
        gpu!(scope, p_b *= yw_);

        gpu!(scope, index = index_base);
        gpu!(scope, index += y1_stride);
        gpu!(scope, index += x0_stride);
        gpu!(scope, p_c = input[index]);
        gpu!(scope, p_c *= xw_);
        gpu!(scope, p_c *= yw);

        gpu!(scope, index = index_base);
        gpu!(scope, index += y1_stride);
        gpu!(scope, index += x1_stride);
        gpu!(scope, p_d = input[index]);
        gpu!(scope, p_d *= xw);
        gpu!(scope, p_d *= yw);

        let sum = scope.create_local(input.item());
        gpu!(scope, sum = p_a + p_b);
        gpu!(scope, sum += p_c);
        gpu!(scope, sum += p_d);
        gpu!(scope, output[id] = sum);
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for InterpolateBilinearEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        InterpolateBilinearShader { input, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>())
    }
}

pub(crate) fn interpolate_bilinear_launch<R: Runtime, E: JitElement>(
    input: JitTensor<R, E, 4>,
    output: JitTensor<R, E, 4>,
) -> JitTensor<R, E, 4> {
    let kernel = InterpolateBilinearEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client)
        .inputs(&[EagerHandle::<R>::new(
            &input.handle,
            &input.strides,
            &input.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
