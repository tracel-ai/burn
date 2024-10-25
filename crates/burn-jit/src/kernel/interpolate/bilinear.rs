use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

use crate::{kernel::Kernel, tensor::JitTensor, JitElement, JitRuntime};

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
        let id = Variable::AbsolutePos;

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

        cpa!(scope, input_stride_0 = stride(input, 0u32));
        cpa!(scope, input_stride_1 = stride(input, 1u32));
        cpa!(scope, input_stride_2 = stride(input, 2u32));
        cpa!(scope, input_stride_3 = stride(input, 3u32));

        cpa!(scope, input_shape_2 = shape(input, 2u32));
        cpa!(scope, input_shape_3 = shape(input, 3u32));

        cpa!(scope, output_stride_0 = stride(output, 0u32));
        cpa!(scope, output_stride_1 = stride(output, 1u32));
        cpa!(scope, output_stride_2 = stride(output, 2u32));
        cpa!(scope, output_stride_3 = stride(output, 3u32));

        cpa!(scope, output_shape_0 = shape(output, 0u32));
        cpa!(scope, output_shape_1 = shape(output, 1u32));
        cpa!(scope, output_shape_2 = shape(output, 2u32));
        cpa!(scope, output_shape_3 = shape(output, 3u32));

        let b = scope.create_local(Elem::UInt);
        let c = scope.create_local(Elem::UInt);
        let h = scope.create_local(Elem::UInt);
        let w = scope.create_local(Elem::UInt);

        cpa!(scope, b = id / output_stride_0);
        cpa!(scope, b = b % output_shape_0);

        cpa!(scope, c = id / output_stride_1);
        cpa!(scope, c = c % output_shape_1);

        cpa!(scope, h = id / output_stride_2);
        cpa!(scope, h = h % output_shape_2);

        cpa!(scope, w = id / output_stride_3);
        cpa!(scope, w = w % output_shape_3);

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

        cpa!(scope, numerator_int = input_shape_2 - 1u32);
        cpa!(scope, denominator_int = output_shape_2 - 1u32);
        cpa!(scope, denominator_int = max(denominator_int, 1u32));
        cpa!(scope, factor_float = cast(h));
        cpa!(scope, numerator_float = cast(numerator_int));
        cpa!(scope, denominator_float = cast(denominator_int));
        cpa!(scope, frac = factor_float * numerator_float);
        cpa!(scope, frac = frac / denominator_float);
        cpa!(scope, v0 = floor(frac));
        cpa!(scope, v1 = ceil(frac));
        cpa!(scope, yw = frac - v0);
        cpa!(scope, yw_ = one - yw);
        cpa!(scope, y0 = cast(v0));
        cpa!(scope, y1 = cast(v1));

        cpa!(scope, numerator_int = input_shape_3 - 1u32);
        cpa!(scope, denominator_int = output_shape_3 - 1u32);
        cpa!(scope, denominator_int = max(denominator_int, 1u32));
        cpa!(scope, factor_float = cast(w));
        cpa!(scope, numerator_float = cast(numerator_int));
        cpa!(scope, denominator_float = cast(denominator_int));
        cpa!(scope, frac = factor_float * numerator_float);
        cpa!(scope, frac = frac / denominator_float);
        cpa!(scope, v0 = floor(frac));
        cpa!(scope, v1 = ceil(frac));
        cpa!(scope, xw = frac - v0);
        cpa!(scope, xw_ = one - xw);
        cpa!(scope, x0 = cast(v0));
        cpa!(scope, x1 = cast(v1));

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

        cpa!(scope, index_base = b * input_stride_0);
        cpa!(scope, index_tmp = c * input_stride_1);
        cpa!(scope, index_base += index_tmp);
        cpa!(scope, y0_stride = y0 * input_stride_2);
        cpa!(scope, y1_stride = y1 * input_stride_2);
        cpa!(scope, x0_stride = x0 * input_stride_3);
        cpa!(scope, x1_stride = x1 * input_stride_3);

        cpa!(scope, index = index_base);
        cpa!(scope, index += y0_stride);
        cpa!(scope, index += x0_stride);
        cpa!(scope, p_a = input[index]);
        cpa!(scope, p_a *= xw_);
        cpa!(scope, p_a *= yw_);

        cpa!(scope, index = index_base);
        cpa!(scope, index += y0_stride);
        cpa!(scope, index += x1_stride);
        cpa!(scope, p_b = input[index]);
        cpa!(scope, p_b *= xw);
        cpa!(scope, p_b *= yw_);

        cpa!(scope, index = index_base);
        cpa!(scope, index += y1_stride);
        cpa!(scope, index += x0_stride);
        cpa!(scope, p_c = input[index]);
        cpa!(scope, p_c *= xw_);
        cpa!(scope, p_c *= yw);

        cpa!(scope, index = index_base);
        cpa!(scope, index += y1_stride);
        cpa!(scope, index += x1_stride);
        cpa!(scope, p_d = input[index]);
        cpa!(scope, p_d *= xw);
        cpa!(scope, p_d *= yw);

        let sum = scope.create_local(input.item());
        cpa!(scope, sum = p_a + p_b);
        cpa!(scope, sum += p_c);
        cpa!(scope, sum += p_d);
        cpa!(scope, output[id] = sum);
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for InterpolateBilinearEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let input = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        InterpolateBilinearShader { input, output }.expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let out = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![input],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>()
    }
}

pub(crate) fn interpolate_bilinear_launch<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    output: JitTensor<R, E>,
) -> JitTensor<R, E> {
    let kernel = InterpolateBilinearEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client.clone())
        .inputs(&[input.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
