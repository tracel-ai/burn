use std::marker::PhantomData;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{cube_inline, ComputeShader, Elem, Scope, Variable, Visibility},
    kernel::GpuComputeShaderPhase,
    tensor::JitTensor,
    JitElement, Runtime,
};

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
        let id = Variable::Id;
        let elem = E::gpu_elem();

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

        cube_inline!(scope, input_stride_0 = stride(input, 0u32));
        cube_inline!(scope, input_stride_1 = stride(input, 1u32));
        cube_inline!(scope, input_stride_2 = stride(input, 2u32));
        cube_inline!(scope, input_stride_3 = stride(input, 3u32));

        cube_inline!(scope, input_shape_2 = shape(input, 2u32));
        cube_inline!(scope, input_shape_3 = shape(input, 3u32));

        cube_inline!(scope, output_stride_0 = stride(output, 0u32));
        cube_inline!(scope, output_stride_1 = stride(output, 1u32));
        cube_inline!(scope, output_stride_2 = stride(output, 2u32));
        cube_inline!(scope, output_stride_3 = stride(output, 3u32));

        cube_inline!(scope, output_shape_0 = shape(output, 0u32));
        cube_inline!(scope, output_shape_1 = shape(output, 1u32));
        cube_inline!(scope, output_shape_2 = shape(output, 2u32));
        cube_inline!(scope, output_shape_3 = shape(output, 3u32));

        let b = scope.create_local(Elem::UInt);
        let c = scope.create_local(Elem::UInt);
        let h = scope.create_local(Elem::UInt);
        let w = scope.create_local(Elem::UInt);

        cube_inline!(scope, b = id / output_stride_0);
        cube_inline!(scope, b = b % output_shape_0);

        cube_inline!(scope, c = id / output_stride_1);
        cube_inline!(scope, c = c % output_shape_1);

        cube_inline!(scope, h = id / output_stride_2);
        cube_inline!(scope, h = h % output_shape_2);

        cube_inline!(scope, w = id / output_stride_3);
        cube_inline!(scope, w = w % output_shape_3);

        let factor_float = scope.create_local(elem);
        let numerator_float = scope.create_local(elem);
        let denominator_float = scope.create_local(elem);
        let x = scope.create_local(elem);
        let y = scope.create_local(elem);
        let xu = scope.create_local(Elem::UInt);
        let yu = scope.create_local(Elem::UInt);

        cube_inline!(scope, factor_float = cast(h));
        cube_inline!(scope, numerator_float = cast(input_shape_2));
        cube_inline!(scope, denominator_float = cast(output_shape_2));
        cube_inline!(scope, y = factor_float * numerator_float);
        cube_inline!(scope, y = y / denominator_float);
        cube_inline!(scope, y = floor(y));
        cube_inline!(scope, yu = cast(y));

        cube_inline!(scope, factor_float = cast(w));
        cube_inline!(scope, numerator_float = cast(input_shape_3));
        cube_inline!(scope, denominator_float = cast(output_shape_3));
        cube_inline!(scope, x = factor_float * numerator_float);
        cube_inline!(scope, x = x / denominator_float);
        cube_inline!(scope, x = floor(x));
        cube_inline!(scope, xu = cast(x));

        let index = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        let val = scope.create_local(output.item());

        cube_inline!(scope, index = b * input_stride_0);
        cube_inline!(scope, index_tmp = c * input_stride_1);
        cube_inline!(scope, index += index_tmp);
        cube_inline!(scope, index_tmp = yu * input_stride_2);
        cube_inline!(scope, index += index_tmp);
        cube_inline!(scope, index_tmp = xu * input_stride_3);
        cube_inline!(scope, index += index_tmp);

        cube_inline!(scope, val = input[index]);
        cube_inline!(scope, output[id] = val);
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for InterpolateNearestEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

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

pub(crate) fn interpolate_nearest_launch<R: Runtime, E: JitElement>(
    input: JitTensor<R, E, 4>,
    output: JitTensor<R, E, 4>,
) -> JitTensor<R, E, 4> {
    let kernel = InterpolateNearestEagerKernel::<R, E>::new();

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
