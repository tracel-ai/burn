use std::marker::PhantomData;

use crate::{
    codegen::{Compilation, CompilationInfo, CompilationSettings, InputInfo, OutputInfo},
    gpu::{gpu, Elem, Scope, Variable, Visibility},
    kernel::{DynamicKernelSource, SourceTemplate},
    Compiler, JitElement, Runtime,
};

pub(crate) struct AdaptivePool2dComputeShader<R: Runtime, E: JitElement> {
    input: Variable,
    output: Variable,
    _elem: PhantomData<E>,
    _runtime: PhantomData<R>,
}

impl<R: Runtime, E: JitElement> AdaptivePool2dComputeShader<R, E> {
    fn expand(self, scope: &mut Scope) {
        let input = self.input;
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

        gpu!(scope, input_shape_0 = shape(input, 2u32));
        gpu!(scope, input_shape_1 = shape(input, 3u32));
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
        let oh = scope.create_local(Elem::UInt);
        let ow = scope.create_local(Elem::UInt);

        gpu!(scope, b = id / output_stride_0);
        gpu!(scope, b = b % output_shape_0);

        gpu!(scope, c = id / output_stride_1);
        gpu!(scope, c = c % output_shape_1);

        gpu!(scope, oh = id / output_stride_2);
        gpu!(scope, oh = oh % output_shape_2);

        gpu!(scope, ow = id / output_stride_3);
        gpu!(scope, ow = ow % output_shape_3);

        let ih_start = scope.create_local(Elem::UInt);
        let ih_end = scope.create_local(Elem::UInt);
        let iw_start = scope.create_local(Elem::UInt);
        let iw_end = scope.create_local(Elem::UInt);
        // TODO COMPUTE THEM ^

        let result = scope.create_local(input.item());

        let index_input = scope.create_local(Elem::UInt);
        let index_input_0 = scope.create_local(Elem::UInt);
        let index_input_1 = scope.create_local(Elem::UInt);
        let index_input_2 = scope.create_local(Elem::UInt);
        let index_input_3 = scope.create_local(Elem::UInt);

        gpu!(scope, index_input_0 = b * input_stride_0);
        gpu!(scope, index_input_1 = c * input_stride_1);

        let sum = scope.zero(output.item());

        gpu!(
            scope,
            range(ih_start, ih_end).for_each(|ih, scope| {
                gpu!(
                    scope,
                    range(iw_start, iw_end).for_each(|iw, scope| {
                        gpu!(scope, index_input_2 = ih * input_stride_2);
                        gpu!(scope, index_input_3 = iw * input_stride_3);

                        gpu!(scope, index_input = index_input_0);
                        gpu!(scope, index_input += index_input_1);
                        gpu!(scope, index_input += index_input_2);
                        gpu!(scope, index_input += index_input_3);

                        gpu!(scope, result = input[index_input]);

                        gpu!(scope, sum += result);
                    })
                );
            })
        );

        let count = scope.create_local(Elem::UInt);
        let count_tmp = scope.create_local(Elem::UInt);
        let count_float = scope.create_local(output.item());
        let avg = scope.create_local(output.item());

        gpu!(scope, count = ih_end - ih_start);
        gpu!(scope, count_tmp = iw_end - iw_start);
        gpu!(scope, count *= count_tmp);

        gpu!(scope, count_float = cast(count));
        gpu!(scope, avg = sum / count_float);
        gpu!(scope, output[id] = sum);
    }
}

#[derive(new)]
pub(crate) struct AdaptivePool2dEagerKernel<R: Runtime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for AdaptivePool2dEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        AdaptivePool2dComputeShader {
            input,
            output,
            _elem: PhantomData::<E>,
            _runtime: PhantomData::<R>,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };

        let output = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>(),)
    }
}
