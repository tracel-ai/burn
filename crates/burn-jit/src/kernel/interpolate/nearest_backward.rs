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
struct InterpolateNearestBackwardEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct InterpolateNearestBackwardShader<E> {
    out_grad: Variable,
    output: Variable,
    _elem: PhantomData<E>,
}

impl<E: JitElement> InterpolateNearestBackwardShader<E> {
    fn expand(self, scope: &mut Scope) {
        let grad = self.out_grad;
        let output = self.output;
        let id = Variable::Id;

        let grad_stride_0 = scope.create_local(Elem::UInt);
        let grad_stride_1 = scope.create_local(Elem::UInt);
        let grad_stride_2 = scope.create_local(Elem::UInt);
        let grad_stride_3 = scope.create_local(Elem::UInt);

        let grad_shape_0 = scope.create_local(Elem::UInt);
        let grad_shape_1 = scope.create_local(Elem::UInt);
        let grad_shape_2 = scope.create_local(Elem::UInt);
        let grad_shape_3 = scope.create_local(Elem::UInt);

        let output_stride_0 = scope.create_local(Elem::UInt);
        let output_stride_1 = scope.create_local(Elem::UInt);
        let output_stride_2 = scope.create_local(Elem::UInt);
        let output_stride_3 = scope.create_local(Elem::UInt);

        let output_shape_0 = scope.create_local(Elem::UInt);
        let output_shape_1 = scope.create_local(Elem::UInt);
        let output_shape_2 = scope.create_local(Elem::UInt);
        let output_shape_3 = scope.create_local(Elem::UInt);

        gpu!(scope, grad_stride_0 = stride(grad, 0u32));
        gpu!(scope, grad_stride_1 = stride(grad, 1u32));
        gpu!(scope, grad_stride_2 = stride(grad, 2u32));
        gpu!(scope, grad_stride_3 = stride(grad, 3u32));

        gpu!(scope, grad_shape_0 = shape(grad, 0u32));
        gpu!(scope, grad_shape_1 = shape(grad, 1u32));
        gpu!(scope, grad_shape_2 = shape(grad, 2u32));
        gpu!(scope, grad_shape_3 = shape(grad, 3u32));

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

        let gh_start = Self::start_index(scope, oh, grad_shape_2, output_shape_2);
        let gh_end = Self::end_index(scope, oh, grad_shape_2, output_shape_2);
        let gw_start = Self::start_index(scope, ow, grad_shape_3, output_shape_3);
        let gw_end = Self::end_index(scope, ow, grad_shape_3, output_shape_3);

        let result = scope.create_local(grad.item());

        let index_grad = scope.create_local(Elem::UInt);
        let index_grad_0 = scope.create_local(Elem::UInt);
        let index_grad_1 = scope.create_local(Elem::UInt);
        let index_grad_2 = scope.create_local(Elem::UInt);
        let index_grad_3 = scope.create_local(Elem::UInt);

        gpu!(scope, index_grad_0 = b * grad_stride_0);
        gpu!(scope, index_grad_1 = c * grad_stride_1);

        let sum = scope.zero(output.item());

        gpu!(
            scope,
            range(gh_start, gh_end).for_each(|gh, scope| {
                gpu!(
                    scope,
                    range(gw_start, gw_end).for_each(|gw, scope| {
                        gpu!(scope, index_grad_2 = gh * grad_stride_2);
                        gpu!(scope, index_grad_3 = gw * grad_stride_3);

                        gpu!(scope, index_grad = index_grad_0);
                        gpu!(scope, index_grad += index_grad_1);
                        gpu!(scope, index_grad += index_grad_2);
                        gpu!(scope, index_grad += index_grad_3);

                        gpu!(scope, result = grad[index_grad]);

                        gpu!(scope, sum += result);
                    })
                );
            })
        );

        gpu!(scope, output[id] = sum);
    }

    fn start_index(
        scope: &mut Scope,
        input_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::gpu_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);

        gpu!(scope, index = input_index * output_size);
        gpu!(scope, numerator_float = cast(index));
        gpu!(scope, div = cast(input_size));
        gpu!(scope, div = numerator_float / div);
        gpu!(scope, div = ceil(div));
        gpu!(scope, index = cast(div));

        index
    }

    fn end_index(
        scope: &mut Scope,
        input_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::gpu_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);
        let min = scope.create_local(Elem::Bool);
        let end_index = scope.create_local(Elem::UInt);

        gpu!(scope, index = input_index + 1u32);
        gpu!(scope, index *= output_size);
        gpu!(scope, numerator_float = cast(index));
        gpu!(scope, div = cast(input_size));
        gpu!(scope, div = numerator_float / div);
        gpu!(scope, div = ceil(div));
        gpu!(scope, index = cast(div));

        gpu!(scope, min = output_size < index);
        gpu!(scope, if(min).then(|scope|{
            gpu!(scope, end_index = output_size);
        }).else(|scope|{
            gpu!(scope, end_index = index);
        }));

        end_index
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase
    for InterpolateNearestBackwardEagerKernel<R, E>
{
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let out_grad = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        InterpolateNearestBackwardShader {
            out_grad,
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

pub(crate) fn interpolate_nearest_backward_launch<R: Runtime, E: JitElement>(
    out_grad: JitTensor<R, E, 4>,
    output: JitTensor<R, E, 4>,
) -> JitTensor<R, E, 4> {
    let kernel = InterpolateNearestBackwardEagerKernel::<R, E>::new();

    Execution::start(kernel, out_grad.client)
        .inputs(&[EagerHandle::<R>::new(
            &out_grad.handle,
            &out_grad.strides,
            &out_grad.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
