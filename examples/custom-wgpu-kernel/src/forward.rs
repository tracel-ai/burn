use crate::FloatTensor;

use super::Backend;
use burn::{
    backend::wgpu::{
        BoolElement, CubeBackend, CubeTensor, FloatElement, IntElement, KernelSource, SourceKernel,
        SourceTemplate, WgpuRuntime, build_info, into_contiguous, kernel_source,
    },
    tensor::Shape,
};
use cubecl::{CubeCount, CubeDim, prelude::KernelId, server::Bindings};
use derive_new::new;
use std::marker::PhantomData;

// Source the kernel written in WGSL.
kernel_source!(FusedMatmulAddReluRaw, "./kernel.wgsl");

// Define our kernel type with cube information.
#[derive(new, Debug)]
struct FusedMatmulAddRelu<E: FloatElement> {
    cube_dim: CubeDim,
    _elem: PhantomData<E>,
}

// Implement the dynamic kernel trait for our kernel type.
impl<E: FloatElement> KernelSource for FusedMatmulAddRelu<E> {
    fn source(&self) -> SourceTemplate {
        // Extend our raw kernel with cube size information using the
        // `SourceTemplate` trait.
        FusedMatmulAddReluRaw::new()
            .source()
            .register("workgroup_size_x", self.cube_dim.x.to_string())
            .register("workgroup_size_y", self.cube_dim.y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.cube_dim)
    }
}

/// Implement our custom backend trait for the existing backend `WgpuBackend`.
impl<F: FloatElement, I: IntElement, BT: BoolElement> Backend
    for CubeBackend<WgpuRuntime, F, I, BT>
{
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        // For simplicity, make sure each tensor is continuous.
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        // Get the matmul relevant shapes.
        let ndims = lhs.shape.num_dims();
        let num_rows = lhs.shape[ndims - 2];
        let num_cols = rhs.shape[ndims - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in shape_out.clone().into_iter().take(ndims - 2) {
            shape_out[i] = usize::max(lhs.shape[i], rhs.shape[i]);
            num_batches *= shape_out[i];
        }
        shape_out[ndims - 2] = num_rows;
        shape_out[ndims - 1] = num_cols;
        let shape_out = Shape::from(shape_out);

        // Create a buffer for the output tensor.
        let buffer = lhs
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Create the kernel.
        let kernel = FusedMatmulAddRelu::<F>::new(cube_dim);

        // Build info buffer with tensor information needed by the kernel, such as shapes and strides.
        let info = build_info::<_, F>(&[&lhs, &rhs, &output]);
        let info_handle = lhs.client.create(bytemuck::cast_slice(&info));

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers.
        lhs.client.execute(
            Box::new(SourceKernel::new(kernel, cube_dim)),
            cube_count,
            Bindings::new().with_buffers(vec![
                lhs.handle.binding(),
                rhs.handle.binding(),
                bias.handle.binding(),
                output.handle.clone().binding(),
                info_handle.binding(),
            ]),
        );

        // Return the output tensor.
        output
    }
}
