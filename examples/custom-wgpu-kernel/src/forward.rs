use crate::FloatTensor;

use super::Backend;
use burn::tensor::Shape;
use burn_wgpu::{
    context::WorkGroup,
    kernel::{build_info, into_contiguous, DynamicKernel, SourceTemplate, StaticKernel},
    kernel_wgsl,
    tensor::WgpuTensor,
    FloatElement, GraphicsApi, IntElement, WgpuBackend,
};
use derive_new::new;
use std::marker::PhantomData;

// Source the kernel written in WGSL.
kernel_wgsl!(FusedMatmulAddReluRaw, "./kernel.wgsl");

// Define our kernel type with workgroup information.
#[derive(new, Debug)]
struct FusedMatmulAddRelu<E: FloatElement> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _elem: PhantomData<E>,
}

// Implement the dynamic kernel trait for our kernel type.
impl<E: FloatElement> DynamicKernel for FusedMatmulAddRelu<E> {
    fn source_template(self) -> SourceTemplate {
        // Extend our raw kernel with workgroup size information using the
        // `SourceTemplate` trait.
        FusedMatmulAddReluRaw::source_template()
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        format!("{:?}", self)
    }
}

/// Implement our custom backend trait for the existing backend `WgpuBackend`.
impl<G: GraphicsApi, F: FloatElement, I: IntElement> Backend for WgpuBackend<G, F, I> {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
        bias: FloatTensor<Self, D>,
    ) -> WgpuTensor<F, D> {
        // Define workgroup size, hardcoded for simplicity.
        let workgroup_size_x = 16;
        let workgroup_size_y = 16;

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        // For simplicity, make sure each tensor is continuous.
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        // Get the matmul relevant shapes.
        let num_rows = lhs.shape.dims[D - 2];
        let num_cols = rhs.shape.dims[D - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = [0; D];
        for i in shape_out.into_iter().take(D - 2) {
            shape_out[i] = usize::max(lhs.shape.dims[i], rhs.shape.dims[i]);
            num_batches *= shape_out[i];
        }
        shape_out[D - 2] = num_rows;
        shape_out[D - 1] = num_cols;
        let shape_out = Shape::new(shape_out);

        // Create a buffer for the output tensor.
        let buffer = lhs
            .context
            .create_buffer(shape_out.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

        let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
        let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;

        // Compile the kernel or use the cache based on the template id.
        let kernel = lhs.context.compile_dynamic(FusedMatmulAddRelu::<F>::new(
            workgroup_size_x,
            workgroup_size_y,
        ));

        // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
        let info = build_info(&[&lhs, &rhs, &output]);
        let info_buffer = lhs
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        // Declare the wgsl workgroup with the number of blocks in x, y and z.
        let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers.
        lhs.context.execute(
            workgroup,
            kernel,
            &[
                &lhs.buffer,
                &rhs.buffer,
                &bias.buffer,
                &output.buffer,
                &info_buffer,
            ],
        );

        // Return the output tensor.
        output
    }
}
