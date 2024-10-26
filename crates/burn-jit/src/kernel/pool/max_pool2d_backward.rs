use crate::{
    element::JitElement,
    kernel::{self, Kernel},
    ops::numeric::empty_device,
    tensor::JitTensor,
    JitRuntime,
};
use cubecl::{
    cpa,
    ir::{Elem, IntKind, Item, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

#[derive(new)]
struct MaxPool2dWithIndicesBackwardEagerKernel<R: JitRuntime, E: JitElement> {
    kernel_size: [usize; 2],
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct MaxPool2dBackwardComputeShader {
    indices: Variable,
    grad: Variable,
    output: Variable,
    kernel_size: [usize; 2],
}

impl MaxPool2dBackwardComputeShader {
    fn expand(self, scope: &mut Scope) {
        let grad = self.grad;
        let output = self.output;
        let indices = self.indices;
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

        let index_current = scope.create_local(Elem::UInt);
        let index_current_tmp = scope.create_local(Elem::UInt);

        cpa!(scope, index_current = ih * output_stride_2);
        cpa!(scope, index_current_tmp = iw * output_stride_3);
        cpa!(scope, index_current += index_current_tmp);

        let index_select = scope.create_local(Elem::Int(IntKind::I32));

        let index_max = scope.create_local(Elem::UInt);
        let is_max = scope.create_local(Elem::Bool);

        let index = scope.create_local(Elem::UInt);
        let index_base = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);

        let grad_accumulation = scope.zero(grad.item());
        let result = scope.create_local(grad.item());

        let (oh_start, oh_end, ow_start, ow_end) = self.loop_ranges(
            scope,
            ih,
            iw,
            grad_shape_2,
            grad_shape_3,
            output_stride_2,
            output_stride_3,
        );

        cpa!(scope, index_base = b * grad_stride_0);
        cpa!(scope, index_tmp = c * grad_stride_1);
        cpa!(scope, index_base += index_tmp);

        cpa!(
            scope,
            range(oh_start, oh_end).for_each(|oh, scope| {
                cpa!(
                    scope,
                    range(ow_start, ow_end).for_each(|ow, scope| {
                        cpa!(scope, index = index_base);
                        cpa!(scope, index_tmp = oh * grad_stride_2);
                        cpa!(scope, index += index_tmp);
                        cpa!(scope, index_tmp = ow * grad_stride_3);
                        cpa!(scope, index += index_tmp);

                        cpa!(scope, index_select = indices[index]);
                        cpa!(scope, index_max = cast(index_select));

                        cpa!(scope, is_max = index_max == index_current);

                        cpa!(scope, if(is_max).then(|scope|{
                            cpa!(scope, result = grad[index]);
                            cpa!(scope, grad_accumulation += result);
                        }));
                    })
                );
            })
        );

        cpa!(scope, output[id] = grad_accumulation);
    }

    #[allow(clippy::too_many_arguments)]
    fn loop_ranges(
        self,
        scope: &mut Scope,
        ih: Variable,
        iw: Variable,
        grad_shape_2: Variable,
        grad_shape_3: Variable,
        output_stride_2: Variable,
        output_stride_3: Variable,
    ) -> (Variable, Variable, Variable, Variable) {
        let pool_stride_0 = Variable::GlobalScalar {
            id: 0,
            elem: Elem::UInt,
        };
        let pool_stride_1 = Variable::GlobalScalar {
            id: 1,
            elem: Elem::UInt,
        };
        let dilation_0 = Variable::GlobalScalar {
            id: 2,
            elem: Elem::UInt,
        };
        let dilation_1 = Variable::GlobalScalar {
            id: 3,
            elem: Elem::UInt,
        };
        let padding_0 = Variable::GlobalScalar {
            id: 4,
            elem: Elem::UInt,
        };
        let padding_1 = Variable::GlobalScalar {
            id: 5,
            elem: Elem::UInt,
        };

        let [kernel_size_0, kernel_size_1] = self.kernel_size;

        let signed_ih = scope.create_local(Elem::Int(IntKind::I32));
        let signed_iw = scope.create_local(Elem::Int(IntKind::I32));

        let signed_pool_stride_0 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_pool_stride_1 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_dilation_0 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_dilation_1 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_padding_0 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_padding_1 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_kernel_size_0 = scope.create_local(Elem::Int(IntKind::I32));
        let signed_kernel_size_1 = scope.create_local(Elem::Int(IntKind::I32));

        cpa!(scope, signed_pool_stride_0 = cast(pool_stride_0));
        cpa!(scope, signed_pool_stride_1 = cast(pool_stride_1));
        cpa!(scope, signed_dilation_0 = cast(dilation_0));
        cpa!(scope, signed_dilation_1 = cast(dilation_1));
        cpa!(scope, signed_padding_0 = cast(padding_0));
        cpa!(scope, signed_padding_1 = cast(padding_1));

        cpa!(scope, signed_kernel_size_0 = cast(kernel_size_0));
        cpa!(scope, signed_kernel_size_1 = cast(kernel_size_1));

        cpa!(scope, signed_ih = cast(ih));
        cpa!(scope, signed_iw = cast(iw));

        let kms_0 = scope.create_local(Elem::Int(IntKind::I32));
        let kms_1 = scope.create_local(Elem::Int(IntKind::I32));

        cpa!(scope, kms_0 = signed_dilation_0 * signed_kernel_size_0);
        cpa!(scope, kms_0 = kms_0 - signed_pool_stride_0);

        cpa!(scope, kms_1 = signed_dilation_1 * signed_kernel_size_1);
        cpa!(scope, kms_1 = kms_1 - signed_pool_stride_1);

        let oh_start_tmp = scope.create_local(Elem::Int(IntKind::I32));
        let ow_start_tmp = scope.create_local(Elem::Int(IntKind::I32));

        cpa!(scope, oh_start_tmp = signed_ih + signed_padding_0);
        cpa!(scope, oh_start_tmp = oh_start_tmp - kms_0);
        cpa!(scope, oh_start_tmp = oh_start_tmp / signed_pool_stride_0);

        cpa!(scope, ow_start_tmp = signed_iw + signed_padding_1);
        cpa!(scope, ow_start_tmp = ow_start_tmp - kms_1);
        cpa!(scope, ow_start_tmp = ow_start_tmp / signed_pool_stride_1);

        cpa!(scope, oh_start_tmp = max(oh_start_tmp, 0i32));
        cpa!(scope, ow_start_tmp = max(ow_start_tmp, 0i32));

        let oh_start = scope.create_local(Elem::UInt);
        let ow_start = scope.create_local(Elem::UInt);

        cpa!(scope, oh_start = cast(oh_start_tmp));
        cpa!(scope, ow_start = cast(ow_start_tmp));

        let oh_end_tmp = scope.create_local(Elem::Int(IntKind::I32));
        let ow_end_tmp = scope.create_local(Elem::Int(IntKind::I32));

        cpa!(scope, oh_end_tmp = max(kms_0, 0i32));
        cpa!(scope, ow_end_tmp = max(kms_1, 0i32));

        let oh_end = scope.create_local(Elem::UInt);
        let ow_end = scope.create_local(Elem::UInt);

        let oh_end_limit = scope.create_local(Elem::UInt);
        let ow_end_limit = scope.create_local(Elem::UInt);

        cpa!(scope, oh_end = cast(oh_end_tmp));
        cpa!(scope, ow_end = cast(ow_end_tmp));

        cpa!(scope, oh_end = oh_end + oh_start);
        cpa!(scope, oh_end_limit = grad_shape_2 - 1u32);

        cpa!(scope, ow_end = ow_end + ow_start);
        cpa!(scope, ow_end_limit = grad_shape_3 - 1u32);

        cpa!(scope, oh_end = min(oh_end, oh_end_limit));
        cpa!(scope, ow_end = min(ow_end, ow_end_limit));

        let index_current = scope.create_local(Elem::UInt);
        let index_current_tmp = scope.create_local(Elem::UInt);

        cpa!(scope, index_current = ih * output_stride_2);
        cpa!(scope, index_current_tmp = iw * output_stride_3);
        cpa!(scope, index_current += index_current_tmp);

        cpa!(scope, oh_end = oh_end + 1u32);
        cpa!(scope, ow_end = ow_end + 1u32);

        (oh_start, oh_end, ow_start, ow_end)
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for MaxPool2dWithIndicesBackwardEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let indices = Variable::GlobalInputArray {
            id: 0,
            item: Item::new(Elem::Int(IntKind::I32)),
        };
        let grad = Variable::GlobalInputArray { id: 1, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        scope.write_global_custom(output);

        MaxPool2dBackwardComputeShader {
            indices,
            grad,
            output,
            kernel_size: self.kernel_size,
        }
        .expand(&mut scope);

        let indices = InputInfo::Array {
            item: Item::new(Elem::Int(IntKind::I32)),
            visibility: Visibility::Read,
        };

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
            inputs: vec![indices, grad, scalars],
            outputs: vec![output],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info(self.kernel_size)
    }
}

pub(crate) fn max_pool2d_with_indices_backward<R: JitRuntime, E: JitElement, I: JitElement>(
    x: JitTensor<R, E>,
    grad: JitTensor<R, E>,
    indices: JitTensor<R, I>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> JitTensor<R, E> {
    let grad = kernel::into_contiguous(grad);
    let indices = kernel::into_contiguous(indices);

    let output = empty_device(x.client.clone(), x.device.clone(), x.shape.clone());
    let kernel = MaxPool2dWithIndicesBackwardEagerKernel::<R, E>::new(kernel_size);

    Execution::start(kernel, x.client.clone())
        .inputs(&[indices.as_handle_ref(), grad.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .with_scalars(&[
            stride[0] as i32,
            stride[1] as i32,
            dilation[0] as i32,
            dilation[1] as i32,
            padding[0] as i32,
            padding[1] as i32,
        ])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
