use std::marker::PhantomData;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::{gpu, ComputeShader, Elem, Scope, Variable, Visibility},
    kernel::GpuComputeShaderPhase,
    tensor::JitTensor,
    Runtime,
};

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
        let id = Variable::Id;

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

        gpu!(scope, grad_stride_0 = stride(grad, 0u32));
        gpu!(scope, grad_stride_1 = stride(grad, 1u32));
        gpu!(scope, grad_stride_2 = stride(grad, 2u32));
        gpu!(scope, grad_stride_3 = stride(grad, 3u32));

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
        let ih = scope.create_local(Elem::UInt);
        let iw = scope.create_local(Elem::UInt);

        gpu!(scope, b = id / output_stride_0);
        gpu!(scope, b = b % output_shape_0);

        gpu!(scope, c = id / output_stride_1);
        gpu!(scope, c = c % output_shape_1);

        gpu!(scope, ih = id / output_stride_2);
        gpu!(scope, ih = ih % output_shape_2);

        gpu!(scope, iw = id / output_stride_3);
        gpu!(scope, iw = iw % output_shape_3);

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
        gpu!(scope, index_base = b * grad_stride_0);
        gpu!(scope, index_tmp = c * grad_stride_1);
        gpu!(scope, index_base += index_tmp);

        gpu!(
            scope,
            range(oh_start, oh_end).for_each(|oh, scope| {
                let ih_start = Self::start_index(scope, oh, grad_shape_2, output_shape_2);
                let ih_end = Self::end_index(scope, oh, grad_shape_2, output_shape_2);
                gpu!(scope, contributed_h = ih >= ih_start);
                gpu!(scope, contributed_tmp = ih < ih_end);
                gpu!(scope, contributed_h = contributed_h && contributed_tmp);

                gpu!(scope, if(contributed_h).then(|scope|{
                    gpu!(
                        scope,
                        range(ow_start, ow_end).for_each(|ow, scope| {
                            let iw_start = Self::start_index(scope, ow, grad_shape_3, output_shape_3);
                            let iw_end = Self::end_index(scope, ow, grad_shape_3, output_shape_3);

                            gpu!(scope, contributed_w = iw >= iw_start);
                            gpu!(scope, contributed_tmp = iw < iw_end);
                            gpu!(scope, contributed_w = contributed_w && contributed_tmp);


                            gpu!(scope, if(contributed_w).then(|scope|{
                                gpu!(scope, count = ih_end - ih_start);
                                gpu!(scope, count_tmp = iw_end - iw_start);
                                gpu!(scope, count *= count_tmp);
                                gpu!(scope, count_float = cast(count));

                                gpu!(scope, index = index_base);
                                gpu!(scope, index_tmp = oh * grad_stride_2);
                                gpu!(scope, index += index_tmp);
                                gpu!(scope, index_tmp = ow * grad_stride_3);
                                gpu!(scope, index += index_tmp);

                                gpu!(scope, the_grad = grad[index]);
                                gpu!(scope, avg = the_grad / count_float);
                                gpu!(scope, grad_acc += avg);
                            }));
                        })
                    );
                }));
            })
        );

        gpu!(scope, output[id] = grad_acc);
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

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase
    for AdaptiveAvgPool2dBackwardEagerKernel<R, E>
{
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let grad = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

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

        let info = CompilationInfo {
            inputs: vec![grad, scalars],
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

pub(crate) fn adaptive_avg_pool2d_backward<R: Runtime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    out_grad: JitTensor<R, E, 4>,
) -> JitTensor<R, E, 4> {
    let output_shape = x.shape.clone();
    let num_elems = output_shape.num_elements();
    let output_buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(
        x.client.clone(),
        x.device.clone(),
        output_shape,
        output_buffer,
    );

    let kernel = AdaptiveAvgPool2dBackwardEagerKernel::<R, E>::new();

    Execution::start(kernel, x.client)
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
