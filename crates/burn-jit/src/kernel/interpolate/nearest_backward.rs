use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

use crate::{kernel::Kernel, tensor::JitTensor, JitElement, JitRuntime};

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
        let id = Variable::AbsolutePos;

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

        cpa!(scope, grad_stride_0 = stride(grad, 0u32));
        cpa!(scope, grad_stride_1 = stride(grad, 1u32));
        cpa!(scope, grad_stride_2 = stride(grad, 2u32));
        cpa!(scope, grad_stride_3 = stride(grad, 3u32));

        cpa!(scope, grad_shape_0 = shape(grad, 0u32));
        cpa!(scope, grad_shape_1 = shape(grad, 1u32));
        cpa!(scope, grad_shape_2 = shape(grad, 2u32));
        cpa!(scope, grad_shape_3 = shape(grad, 3u32));

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
        let oh = scope.create_local(Elem::UInt);
        let ow = scope.create_local(Elem::UInt);

        cpa!(scope, b = id / output_stride_0);
        cpa!(scope, b = b % output_shape_0);

        cpa!(scope, c = id / output_stride_1);
        cpa!(scope, c = c % output_shape_1);

        cpa!(scope, oh = id / output_stride_2);
        cpa!(scope, oh = oh % output_shape_2);

        cpa!(scope, ow = id / output_stride_3);
        cpa!(scope, ow = ow % output_shape_3);

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

        cpa!(scope, index_grad_0 = b * grad_stride_0);
        cpa!(scope, index_grad_1 = c * grad_stride_1);

        let sum = scope.zero(output.item());

        cpa!(
            scope,
            range(gh_start, gh_end).for_each(|gh, scope| {
                cpa!(
                    scope,
                    range(gw_start, gw_end).for_each(|gw, scope| {
                        cpa!(scope, index_grad_2 = gh * grad_stride_2);
                        cpa!(scope, index_grad_3 = gw * grad_stride_3);

                        cpa!(scope, index_grad = index_grad_0);
                        cpa!(scope, index_grad += index_grad_1);
                        cpa!(scope, index_grad += index_grad_2);
                        cpa!(scope, index_grad += index_grad_3);

                        cpa!(scope, result = grad[index_grad]);

                        cpa!(scope, sum += result);
                    })
                );
            })
        );

        cpa!(scope, output[id] = sum);
    }

    fn start_index(
        scope: &mut Scope,
        input_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::cube_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);

        cpa!(scope, index = input_index * output_size);
        cpa!(scope, numerator_float = cast(index));
        cpa!(scope, div = cast(input_size));
        cpa!(scope, div = numerator_float / div);
        cpa!(scope, div = ceil(div));
        cpa!(scope, index = cast(div));

        index
    }

    fn end_index(
        scope: &mut Scope,
        input_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::cube_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);
        let min = scope.create_local(Elem::Bool);
        let end_index = scope.create_local(Elem::UInt);

        cpa!(scope, index = input_index + 1u32);
        cpa!(scope, index *= output_size);
        cpa!(scope, numerator_float = cast(index));
        cpa!(scope, div = cast(input_size));
        cpa!(scope, div = numerator_float / div);
        cpa!(scope, div = ceil(div));
        cpa!(scope, index = cast(div));

        cpa!(scope, min = output_size < index);
        cpa!(scope, if(min).then(|scope|{
            cpa!(scope, end_index = output_size);
        }).else(|scope|{
            cpa!(scope, end_index = index);
        }));

        end_index
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for InterpolateNearestBackwardEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let out_grad = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

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

pub(crate) fn interpolate_nearest_backward_launch<R: JitRuntime, E: JitElement>(
    out_grad: JitTensor<R, E>,
    output: JitTensor<R, E>,
) -> JitTensor<R, E> {
    let kernel = InterpolateNearestBackwardEagerKernel::<R, E>::new();

    Execution::start(kernel, out_grad.client.clone())
        .inputs(&[out_grad.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
