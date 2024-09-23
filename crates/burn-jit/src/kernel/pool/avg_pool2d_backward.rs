use crate::{
    element::JitElement,
    kernel::{self, Kernel},
    ops::numeric::empty_device,
    tensor::JitTensor,
    JitRuntime,
};
use cubecl::{
    cpa,
    ir::{Elem, IntKind, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

#[derive(new)]
struct AvgPool2dBackwardEagerKernel<R: JitRuntime, E: JitElement> {
    kernel_size: [usize; 2],
    count_include_pad: bool,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct AvgPool2dBackwardComputeShader {
    grad: Variable,
    output: Variable,
    kernel_size: [usize; 2],
    count_include_pad: bool,
}

impl AvgPool2dBackwardComputeShader {
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

        let pool_stride_0 = Variable::GlobalScalar {
            id: 0,
            elem: Elem::UInt,
        };
        let pool_stride_1 = Variable::GlobalScalar {
            id: 1,
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

        let index = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        let index_base = scope.create_local(Elem::UInt);

        let grad_accumulation = scope.zero(grad.item());
        let result = scope.create_local(grad.item());
        let count = scope.create_local(grad.item());

        let count_include_pad = self.count_include_pad;
        if count_include_pad {
            let kernel_size: Variable = (self.kernel_size[0] * self.kernel_size[1]).into();
            cpa!(scope, count = kernel_size);
        }

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

        let border_bottom = scope.create_local(Elem::UInt);
        let border_right = scope.create_local(Elem::UInt);
        let begin_h = scope.create_local(Elem::UInt);
        let begin_w = scope.create_local(Elem::UInt);
        let iw_start = scope.create_local(Elem::UInt);
        let iw_end = scope.create_local(Elem::UInt);
        let ih_start = scope.create_local(Elem::UInt);
        let ih_end = scope.create_local(Elem::UInt);
        let after_start = scope.create_local(Elem::Bool);
        let before_end = scope.create_local(Elem::Bool);
        let contributed_h = scope.create_local(Elem::Bool);
        let contributed_w = scope.create_local(Elem::Bool);
        cpa!(scope, border_bottom = output_shape_2 + padding_0);
        cpa!(scope, border_right = output_shape_3 + padding_1);
        cpa!(scope, begin_h = ih + padding_0);
        cpa!(scope, begin_w = iw + padding_1);

        let ih_diff = scope.create_local(Elem::UInt);
        let iw_diff = scope.create_local(Elem::UInt);
        let count_int = scope.create_local(Elem::UInt);

        cpa!(
            scope,
            range(oh_start, oh_end).for_each(|oh, scope| {
                // Contributed h
                cpa!(scope, ih_start = oh * pool_stride_0);
                cpa!(scope, ih_end = ih_start + kernel_size_0);
                cpa!(scope, ih_start = max(ih_start, padding_0));
                cpa!(scope, ih_end = min(ih_end, border_bottom));
                cpa!(scope, after_start = begin_h >= ih_start);
                cpa!(scope, before_end = ih < ih_end);
                cpa!(scope, contributed_h = after_start && before_end);

                if !count_include_pad {
                    cpa!(scope, ih_diff = ih_end - ih_start);
                }

                cpa!(scope, if(contributed_h).then(|scope|{
                cpa!(
                    scope,
                    range(ow_start, ow_end).for_each(|ow, scope| {
                        cpa!(scope, index = index_base);
                        cpa!(scope, index_tmp = oh * grad_stride_2);
                        cpa!(scope, index += index_tmp);
                        cpa!(scope, index_tmp = ow * grad_stride_3);
                        cpa!(scope, index += index_tmp);

                        // Contributed w
                        cpa!(scope, iw_start = ow * pool_stride_1);
                        cpa!(scope, iw_end = iw_start + kernel_size_1);
                        cpa!(scope, iw_start = max(iw_start, padding_1));
                        cpa!(scope, iw_end = min(iw_end, border_right));
                        cpa!(scope, after_start = begin_w >= iw_start);
                        cpa!(scope, before_end = iw < iw_end);
                        cpa!(scope, contributed_w = after_start && before_end);

                        cpa!(scope, if(contributed_w).then(|scope|{
                            if !count_include_pad {
                                cpa!(scope, iw_diff = iw_end - iw_start);
                                cpa!(scope, count_int = ih_diff * iw_diff);
                                cpa!(scope, count = cast(count_int));
                            }

                            cpa!(scope, result = grad[index]);
                            cpa!(scope, result = result / count);
                            cpa!(scope, grad_accumulation += result);
                        }));
                    }));
                }));
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

impl<R: JitRuntime, E: JitElement> Kernel for AvgPool2dBackwardEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let grad = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        scope.write_global_custom(output);

        AvgPool2dBackwardComputeShader {
            grad,
            output,
            kernel_size: self.kernel_size,
            count_include_pad: self.count_include_pad,
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
        cubecl::KernelId::new::<Self>().info((self.kernel_size, self.count_include_pad))
    }
}

pub(crate) fn avg_pool2d_backward<R: JitRuntime, E: JitElement>(
    x: JitTensor<R, E>,
    grad: JitTensor<R, E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> JitTensor<R, E> {
    let grad = kernel::into_contiguous(grad);
    let dilation = 1;

    let output = empty_device(x.client.clone(), x.device.clone(), x.shape.clone());
    let kernel = AvgPool2dBackwardEagerKernel::<R, E>::new(kernel_size, count_include_pad);

    Execution::start(kernel, x.client)
        .inputs(&[grad.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .with_scalars(&[
            stride[0] as i32,
            stride[1] as i32,
            dilation,
            dilation,
            padding[0] as i32,
            padding[1] as i32,
        ])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
