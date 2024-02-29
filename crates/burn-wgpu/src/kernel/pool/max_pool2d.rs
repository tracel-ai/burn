use crate::{
    codegen::dialect::gpu::{gpu, Elem, Scope, Variable},
    compute::StaticKernel,
    element::JitElement,
    kernel::{
        self, elemwise_workgroup, pool::build_pool2d_info, KernelSettings, WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

kernel_wgsl!(
    MaxPool2dWithIndicesBackward,
    "../../template/pool/max_pool2d_with_indices_backward.wgsl"
);

#[derive(new)]
pub struct MaxPool2dWithIndicesBackwardEagerKernel<R: Runtime, E: JitElement> {
    kernel_size: [usize; 2],
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

pub struct MaxPool2dBackwardComputeShader {
    indices: Variable,
    grad: Variable,
    output: Variable,
    kernel_size: [usize; 2],
}

impl MaxPool2dBackwardComputeShader {
    pub fn expand(self, scope: &mut Scope) {
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
        let dilation_0 = Variable::GlobalScalar(2, Elem::UInt);
        let dilation_1 = Variable::GlobalScalar(3, Elem::UInt);
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

        let kms_0 = scope.create_local(Elem::Int);
        let kms_1 = scope.create_local(Elem::Int);

        gpu!(scope, kms_0 = dilation_0 * kernel_size_0);
        gpu!(scope, kms_0 = kms_0 - pool_stride_0);

        gpu!(scope, kms_1 = dilation_1 * kernel_size_1);
        gpu!(scope, kms_1 = kms_1 + pool_stride_1);

        let oh_start_tmp = scope.create_local(Elem::UInt);
        let ow_start_tmp = scope.create_local(Elem::UInt);

        gpu!(scope, oh_start_tmp = ih + padding_0);
        gpu!(scope, oh_start_tmp = oh_start_tmp - kms_0);
        gpu!(scope, oh_start_tmp = oh_start_tmp / pool_stride_0);

        gpu!(scope, ow_start_tmp = iw + padding_1);
        gpu!(scope, ow_start_tmp = ow_start_tmp - kms_1);
        gpu!(scope, ow_start_tmp = ow_start_tmp / pool_stride_1);
    }
}

pub(crate) fn max_pool2d_with_indices_backward<R: Runtime, E: JitElement, I: JitElement>(
    x: JitTensor<R, E, 4>,
    grad: JitTensor<R, E, 4>,
    indices: JitTensor<R, I, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> JitTensor<R, E, 4> {
    let grad = kernel::into_contiguous(grad);
    let indices = kernel::into_contiguous(indices);

    let num_elems = x.shape.num_elements();
    let buffer = x.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(x.client.clone(), x.device.clone(), x.shape.clone(), buffer);

    let info_handle = build_pool2d_info(&x, &grad, kernel_size, stride, padding, dilation);

    let kernel = StaticKernel::<
        KernelSettings<MaxPool2dWithIndicesBackward, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        output.shape.num_elements(),
        WORKGROUP_DEFAULT,
    ));

    x.client.execute(
        Box::new(kernel),
        &[&indices.handle, &grad.handle, &output.handle, &info_handle],
    );
    output
}
