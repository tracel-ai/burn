use crate::{kernel::fused_matmul_add_relu_kernel, FloatTensor};

use super::Backend;
use burn::tensor::Shape;
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::{CubeCount, CubeDim};

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Backend
    for CubeBackend<R, F, I, BT>
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
        let num_rows = lhs.shape.dims[ndims - 2];
        let num_cols = rhs.shape.dims[ndims - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in shape_out.clone().into_iter().take(ndims - 2) {
            shape_out[i] = usize::max(lhs.shape.dims[i], rhs.shape.dims[i]);
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

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        fused_matmul_add_relu_kernel::launch::<F, R>(
            &lhs.client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<F>(1),
            rhs.as_tensor_arg::<F>(1),
            bias.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        // Return the output tensor.
        output
    }
}
