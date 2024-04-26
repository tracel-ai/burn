use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, IntKind, Scope, Variable, Visibility},
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::ComputeShader,
    kernel::{self, GpuComputeShaderPhase},
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

#[derive(new)]
struct AvgPool2dBackwardEagerKernel<R: Runtime, E: JitElement> {
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

        let pool_stride_0 = Variable::GlobalScalar(0, Elem::UInt);
        let pool_stride_1 = Variable::GlobalScalar(1, Elem::UInt);
        let padding_0 = Variable::GlobalScalar(4, Elem::UInt);
        let padding_1 = Variable::GlobalScalar(5, Elem::UInt);
        let [kernel_size_0, kernel_size_1] = self.kernel_size;

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

        let index_current = scope.create_local(Elem::UInt);
        let index_current_tmp = scope.create_local(Elem::UInt);

        gpu!(scope, index_current = ih * output_stride_2);
        gpu!(scope, index_current_tmp = iw * output_stride_3);
        gpu!(scope, index_current += index_current_tmp);

        let index = scope.create_local(Elem::UInt);
        let index_tmp = scope.create_local(Elem::UInt);
        let index_base = scope.create_local(Elem::UInt);

        let grad_accumulation = scope.zero(grad.item());
        let result = scope.create_local(grad.item());
        let count = scope.create_local(grad.item());

        let count_include_pad = self.count_include_pad;
        if count_include_pad {
            let kernel_size: Variable = (self.kernel_size[0] * self.kernel_size[1]).into();
            gpu!(scope, count = kernel_size);
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

        gpu!(scope, index_base = b * grad_stride_0);
        gpu!(scope, index_tmp = c * grad_stride_1);
        gpu!(scope, index_base += index_tmp);

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
        gpu!(scope, border_bottom = output_shape_2 + padding_0);
        gpu!(scope, border_right = output_shape_3 + padding_1);
        gpu!(scope, begin_h = ih + padding_0);
        gpu!(scope, begin_w = iw + padding_1);

        let ih_diff = scope.create_local(Elem::UInt);
        let iw_diff = scope.create_local(Elem::UInt);
        let count_int = scope.create_local(Elem::UInt);

        gpu!(
            scope,
            range(oh_start, oh_end).for_each(|oh, scope| {
                // Contributed h
                gpu!(scope, ih_start = oh * pool_stride_0);
                gpu!(scope, ih_end = ih_start + kernel_size_0);
                gpu!(scope, ih_start = max(ih_start, padding_0));
                gpu!(scope, ih_end = min(ih_end, border_bottom));
                gpu!(scope, after_start = begin_h >= ih_start);
                gpu!(scope, before_end = ih < ih_end);
                gpu!(scope, contributed_h = after_start && before_end);

                if !count_include_pad {
                    gpu!(scope, ih_diff = ih_end - ih_start);
                }

                gpu!(scope, if(contributed_h).then(|scope|{
                gpu!(
                    scope,
                    range(ow_start, ow_end).for_each(|ow, scope| {
                        gpu!(scope, index = index_base);
                        gpu!(scope, index_tmp = oh * grad_stride_2);
                        gpu!(scope, index += index_tmp);
                        gpu!(scope, index_tmp = ow * grad_stride_3);
                        gpu!(scope, index += index_tmp);

                        // Contributed w
                        gpu!(scope, iw_start = ow * pool_stride_1);
                        gpu!(scope, iw_end = iw_start + kernel_size_1);
                        gpu!(scope, iw_start = max(iw_start, padding_1));
                        gpu!(scope, iw_end = min(iw_end, border_right));
                        gpu!(scope, after_start = begin_w >= iw_start);
                        gpu!(scope, before_end = iw < iw_end);
                        gpu!(scope, contributed_w = after_start && before_end);

                        gpu!(scope, if(contributed_w).then(|scope|{
                            if !count_include_pad {
                                gpu!(scope, iw_diff = iw_end - iw_start);
                                gpu!(scope, count_int = ih_diff * iw_diff);
                                gpu!(scope, count = cast(count_int));
                            }

                            gpu!(scope, result = grad[index]);
                            gpu!(scope, result = result / count);
                            gpu!(scope, grad_accumulation += result);
                        }));
                    }));
                }));
            })
        );

        gpu!(scope, output[id] = grad_accumulation);
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
        let pool_stride_0 = Variable::GlobalScalar(0, Elem::UInt);
        let pool_stride_1 = Variable::GlobalScalar(1, Elem::UInt);
        let dilation_0 = Variable::GlobalScalar(2, Elem::UInt);
        let dilation_1 = Variable::GlobalScalar(3, Elem::UInt);
        let padding_0 = Variable::GlobalScalar(4, Elem::UInt);
        let padding_1 = Variable::GlobalScalar(5, Elem::UInt);

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

        gpu!(scope, signed_pool_stride_0 = cast(pool_stride_0));
        gpu!(scope, signed_pool_stride_1 = cast(pool_stride_1));
        gpu!(scope, signed_dilation_0 = cast(dilation_0));
        gpu!(scope, signed_dilation_1 = cast(dilation_1));
        gpu!(scope, signed_padding_0 = cast(padding_0));
        gpu!(scope, signed_padding_1 = cast(padding_1));

        gpu!(scope, signed_kernel_size_0 = cast(kernel_size_0));
        gpu!(scope, signed_kernel_size_1 = cast(kernel_size_1));

        gpu!(scope, signed_ih = cast(ih));
        gpu!(scope, signed_iw = cast(iw));

        let kms_0 = scope.create_local(Elem::Int(IntKind::I32));
        let kms_1 = scope.create_local(Elem::Int(IntKind::I32));

        gpu!(scope, kms_0 = signed_dilation_0 * signed_kernel_size_0);
        gpu!(scope, kms_0 = kms_0 - signed_pool_stride_0);

        gpu!(scope, kms_1 = signed_dilation_1 * signed_kernel_size_1);
        gpu!(scope, kms_1 = kms_1 - signed_pool_stride_1);

        let oh_start_tmp = scope.create_local(Elem::Int(IntKind::I32));
        let ow_start_tmp = scope.create_local(Elem::Int(IntKind::I32));

        gpu!(scope, oh_start_tmp = signed_ih + signed_padding_0);
        gpu!(scope, oh_start_tmp = oh_start_tmp - kms_0);
        gpu!(scope, oh_start_tmp = oh_start_tmp / signed_pool_stride_0);

        gpu!(scope, ow_start_tmp = signed_iw + signed_padding_1);
        gpu!(scope, ow_start_tmp = ow_start_tmp - kms_1);
        gpu!(scope, ow_start_tmp = ow_start_tmp / signed_pool_stride_1);

        gpu!(scope, oh_start_tmp = max(oh_start_tmp, 0i32));
        gpu!(scope, ow_start_tmp = max(ow_start_tmp, 0i32));

        let oh_start = scope.create_local(Elem::UInt);
        let ow_start = scope.create_local(Elem::UInt);

        gpu!(scope, oh_start = cast(oh_start_tmp));
        gpu!(scope, ow_start = cast(ow_start_tmp));

        let oh_end_tmp = scope.create_local(Elem::Int(IntKind::I32));
        let ow_end_tmp = scope.create_local(Elem::Int(IntKind::I32));

        gpu!(scope, oh_end_tmp = max(kms_0, 0i32));
        gpu!(scope, ow_end_tmp = max(kms_1, 0i32));

        let oh_end = scope.create_local(Elem::UInt);
        let ow_end = scope.create_local(Elem::UInt);

        let oh_end_limit = scope.create_local(Elem::UInt);
        let ow_end_limit = scope.create_local(Elem::UInt);

        gpu!(scope, oh_end = cast(oh_end_tmp));
        gpu!(scope, ow_end = cast(ow_end_tmp));

        gpu!(scope, oh_end = oh_end + oh_start);
        gpu!(scope, oh_end_limit = grad_shape_2 - 1u32);

        gpu!(scope, ow_end = ow_end + ow_start);
        gpu!(scope, ow_end_limit = grad_shape_3 - 1u32);

        gpu!(scope, oh_end = min(oh_end, oh_end_limit));
        gpu!(scope, ow_end = min(ow_end, ow_end_limit));

        let index_current = scope.create_local(Elem::UInt);
        let index_current_tmp = scope.create_local(Elem::UInt);

        gpu!(scope, index_current = ih * output_stride_2);
        gpu!(scope, index_current_tmp = iw * output_stride_3);
        gpu!(scope, index_current += index_current_tmp);

        gpu!(scope, oh_end = oh_end + 1u32);
        gpu!(scope, ow_end = ow_end + 1u32);

        (oh_start, oh_end, ow_start, ow_end)
    }
}

impl<R: Runtime, E: JitElement> GpuComputeShaderPhase for AvgPool2dBackwardEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let grad = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

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

        let info = CompilationInfo {
            inputs: vec![grad, scalars],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!(
            "{:?}k={:?}count_include_pad={:?}",
            core::any::TypeId::of::<Self>(),
            self.kernel_size,
            self.count_include_pad
        )
    }
}

pub(crate) fn avg_pool2d_backward<R: Runtime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    grad: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> JitTensor<R, E, 4> {
    let grad = kernel::into_contiguous(grad);
    let dilation = 1;

    let output = empty_device(x.client.clone(), x.device.clone(), x.shape.clone());
    let kernel = AvgPool2dBackwardEagerKernel::<R, E>::new(kernel_size, count_include_pad);

    Execution::start(kernel, x.client)
        .inputs(&[EagerHandle::<R>::new(
            &grad.handle,
            &grad.strides,
            &grad.shape.dims,
        )])
        .outputs(&[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .with_scalars(&[
            stride[0] as i32,
            stride[1] as i32,
            dilation,
            dilation,
            padding[0] as i32,
            padding[1] as i32,
        ])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
