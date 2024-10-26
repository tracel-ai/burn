use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

use crate::{kernel::Kernel, tensor::JitTensor, JitElement, JitRuntime};

#[derive(new)]
struct InterpolateNearestEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct InterpolateNearestShader<E> {
    input: Variable,
    output: Variable,
    _elem: PhantomData<E>,
}

impl<E: JitElement> InterpolateNearestShader<E> {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::AbsolutePos;
        let elem = E::cube_elem();

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

        let factor_float = scope.create_local(elem);
        let numerator_float = scope.create_local(elem);
        let denominator_float = scope.create_local(elem);
        let x = scope.create_local(elem);
        let y = scope.create_local(elem);
        let xu = scope.create_local(Elem::UInt);
        let yu = scope.create_local(Elem::UInt);

        cpa!(scope, factor_float = cast(h));
        cpa!(scope, numerator_float = cast(input_shape_2));
        cpa!(scope, denominator_float = cast(output_shape_2));
        cpa!(scope, y = factor_float * numerator_float);
        cpa!(scope, y = y / denominator_float);
        cpa!(scope, y = floor(y));
        cpa!(scope, yu = cast(y));

        cpa!(scope, factor_float = cast(w));
        cpa!(scope, numerator_float = cast(input_shape_3));
        cpa!(scope, denominator_float = cast(output_shape_3));
        cpa!(scope, x = factor_float * numerator_float);
        cpa!(scope, x = x / denominator_float);
        cpa!(scope, x = floor(x));
        cpa!(scope, xu = cast(x));

        let index = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        let val = scope.create_local(output.item());

        cpa!(scope, index = b * input_stride_0);
        cpa!(scope, index_tmp = c * input_stride_1);
        cpa!(scope, index += index_tmp);
        cpa!(scope, index_tmp = yu * input_stride_2);
        cpa!(scope, index += index_tmp);
        cpa!(scope, index_tmp = xu * input_stride_3);
        cpa!(scope, index += index_tmp);

        cpa!(scope, val = input[index]);
        cpa!(scope, output[id] = val);
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for InterpolateNearestEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let input = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        InterpolateNearestShader {
            input,
            output,
            _elem: PhantomData::<E>,
        }
        .expand(&mut scope);

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

pub(crate) fn interpolate_nearest_launch<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    output: JitTensor<R, E>,
) -> JitTensor<R, E> {
    let kernel = InterpolateNearestEagerKernel::<R, E>::new();

    Execution::start(kernel, input.client.clone())
        .inputs(&[input.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
