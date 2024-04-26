use std::marker::PhantomData;

use crate::{
    codegen::{Compilation, CompilationInfo, CompilationSettings, InputInfo, OutputInfo},
    gpu::{gpu, ComputeShader, Elem, Scope, Variable, Visibility},
    kernel::GpuComputeShaderPhase,
    JitElement, Runtime,
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

        let ih_start = Self::start_index(scope, oh, output_shape_2, input_shape_2);
        let ih_end = Self::end_index(scope, oh, output_shape_2, input_shape_2);
        let iw_start = Self::start_index(scope, ow, output_shape_3, input_shape_3);
        let iw_end = Self::end_index(scope, ow, output_shape_3, input_shape_3);

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
        gpu!(scope, output[id] = avg);
    }

    fn start_index(
        scope: &mut Scope,
        output_size_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::gpu_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);

        gpu!(scope, index = output_size_index * input_size);
        gpu!(scope, numerator_float = cast(index));
        gpu!(scope, div = cast(output_size));
        gpu!(scope, div = numerator_float / div);
        gpu!(scope, div = floor(div));
        gpu!(scope, index = cast(div));
        index
    }

    fn end_index(
        scope: &mut Scope,
        output_size_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::gpu_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);
        let min = scope.create_local(Elem::Bool);
        let end_index = scope.create_local(Elem::UInt);

        gpu!(scope, index = output_size_index + 1u32);
        gpu!(scope, index *= input_size);
        gpu!(scope, numerator_float = cast(index));
        gpu!(scope, div = cast(output_size));
        gpu!(scope, div = numerator_float / div);
        gpu!(scope, div = ceil(div));
        gpu!(scope, index = cast(div));

        gpu!(scope, min = input_size < index);
        gpu!(scope, if(min).then(|scope|{
            gpu!(scope, end_index = input_size);
        }).else(|scope|{
            gpu!(scope, end_index = index);
        }));
        end_index
    }
}

#[derive(new)]
pub(crate) struct AdaptivePool2dEagerKernel<R: Runtime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for AdaptivePool2dEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
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
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>(),)
    }
}
