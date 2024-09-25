use std::marker::PhantomData;

use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};

use crate::{element::JitElement, kernel::Kernel, tensor::JitTensor, JitRuntime};

#[derive(new)]
struct AdaptiveAvgPool2dBackwardEagerKernel<R, E> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct AdaptiveAvgPool2dBackwardComputeShader<E> {
    grad: Variable,
    output: Variable,
    _elem: PhantomData<E>,
}

impl<E: JitElement> AdaptiveAvgPool2dBackwardComputeShader<E> {
    fn expand(self, scope: &mut Scope) {
        let grad = self.grad;
        let output = self.output;
        let id = Variable::AbsolutePos;

        let grad_stride_0 = scope.create_local(Elem::UInt);
        let grad_stride_1 = scope.create_local(Elem::UInt);
        let grad_stride_2 = scope.create_local(Elem::UInt);
        let grad_stride_3 = scope.create_local(Elem::UInt);

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
        let ih = scope.create_local(Elem::UInt);
        let iw = scope.create_local(Elem::UInt);

        cpa!(scope, b = id / output_stride_0);
        cpa!(scope, b = b % output_shape_0);

        cpa!(scope, c = id / output_stride_1);
        cpa!(scope, c = c % output_shape_1);

        cpa!(scope, ih = id / output_stride_2);
        cpa!(scope, ih = ih % output_shape_2);

        cpa!(scope, iw = id / output_stride_3);
        cpa!(scope, iw = iw % output_shape_3);

        let oh_start = Self::start_index(scope, ih, output_shape_2, grad_shape_2);
        let oh_end = Self::end_index(scope, ih, output_shape_2, grad_shape_2);

        let ow_start = Self::start_index(scope, iw, output_shape_3, grad_shape_3);
        let ow_end = Self::end_index(scope, iw, output_shape_3, grad_shape_3);

        let grad_acc = scope.create_local(output.item());
        let contributed_h = scope.create_local(Elem::Bool);
        let contributed_w = scope.create_local(Elem::Bool);
        let contributed_tmp = scope.create_local(Elem::Bool);

        let count = scope.create_local(Elem::UInt);
        let count_tmp = scope.create_local(Elem::UInt);
        let count_float = scope.create_local(output.item());
        let the_grad = scope.create_local(output.item());
        let avg = scope.create_local(output.item());

        let index_base = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        let index = scope.create_local(Elem::UInt);
        cpa!(scope, index_base = b * grad_stride_0);
        cpa!(scope, index_tmp = c * grad_stride_1);
        cpa!(scope, index_base += index_tmp);

        cpa!(
            scope,
            range(oh_start, oh_end).for_each(|oh, scope| {
                let ih_start = Self::start_index(scope, oh, grad_shape_2, output_shape_2);
                let ih_end = Self::end_index(scope, oh, grad_shape_2, output_shape_2);
                cpa!(scope, contributed_h = ih >= ih_start);
                cpa!(scope, contributed_tmp = ih < ih_end);
                cpa!(scope, contributed_h = contributed_h && contributed_tmp);

                cpa!(scope, if(contributed_h).then(|scope|{
                    cpa!(
                        scope,
                        range(ow_start, ow_end).for_each(|ow, scope| {
                            let iw_start = Self::start_index(scope, ow, grad_shape_3, output_shape_3);
                            let iw_end = Self::end_index(scope, ow, grad_shape_3, output_shape_3);

                            cpa!(scope, contributed_w = iw >= iw_start);
                            cpa!(scope, contributed_tmp = iw < iw_end);
                            cpa!(scope, contributed_w = contributed_w && contributed_tmp);


                            cpa!(scope, if(contributed_w).then(|scope|{
                                cpa!(scope, count = ih_end - ih_start);
                                cpa!(scope, count_tmp = iw_end - iw_start);
                                cpa!(scope, count *= count_tmp);
                                cpa!(scope, count_float = cast(count));

                                cpa!(scope, index = index_base);
                                cpa!(scope, index_tmp = oh * grad_stride_2);
                                cpa!(scope, index += index_tmp);
                                cpa!(scope, index_tmp = ow * grad_stride_3);
                                cpa!(scope, index += index_tmp);

                                cpa!(scope, the_grad = grad[index]);
                                cpa!(scope, avg = the_grad / count_float);
                                cpa!(scope, grad_acc += avg);
                            }));
                        })
                    );
                }));
            })
        );

        cpa!(scope, output[id] = grad_acc);
    }

    fn start_index(
        scope: &mut Scope,
        output_size_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::cube_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);

        cpa!(scope, index = output_size_index * input_size);
        cpa!(scope, numerator_float = cast(index));
        cpa!(scope, div = cast(output_size));
        cpa!(scope, div = numerator_float / div);
        cpa!(scope, div = floor(div));
        cpa!(scope, index = cast(div));
        index
    }

    fn end_index(
        scope: &mut Scope,
        output_size_index: Variable,
        output_size: Variable,
        input_size: Variable,
    ) -> Variable {
        let elem = E::cube_elem();
        let numerator_float = scope.create_local(elem);
        let div = scope.create_local(elem);
        let index = scope.create_local(Elem::UInt);
        let min = scope.create_local(Elem::Bool);
        let end_index = scope.create_local(Elem::UInt);

        cpa!(scope, index = output_size_index + 1u32);
        cpa!(scope, index *= input_size);
        cpa!(scope, numerator_float = cast(index));
        cpa!(scope, div = cast(output_size));
        cpa!(scope, div = numerator_float / div);
        cpa!(scope, div = ceil(div));
        cpa!(scope, index = cast(div));

        cpa!(scope, min = input_size < index);
        cpa!(scope, if(min).then(|scope|{
            cpa!(scope, end_index = input_size);
        }).else(|scope|{
            cpa!(scope, end_index = index);
        }));
        end_index
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for AdaptiveAvgPool2dBackwardEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let grad = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        scope.write_global_custom(output);

        AdaptiveAvgPool2dBackwardComputeShader {
            grad,
            output,
            _elem: PhantomData::<E>,
        }
        .expand(&mut scope);

        let grad = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let scalars = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 6,
        };
        let output = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![grad, scalars],
            outputs: vec![output],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>()
    }
}

pub(crate) fn adaptive_avg_pool2d_backward<R: JitRuntime, E: JitElement>(
    x: JitTensor<R, E>,
    out_grad: JitTensor<R, E>,
) -> JitTensor<R, E> {
    let output_shape = x.shape.clone();
    let num_elems = output_shape.num_elements();
    let output_buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        output_shape,
        output_buffer,
    );

    let kernel = AdaptiveAvgPool2dBackwardEagerKernel::<R, E>::new();

    Execution::start(kernel, x.client)
        .inputs(&[out_grad.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
