use crate::{FloatTensor, kernel::fused_matmul_add_relu_kernel};

use super::Backend;
use burn::{backend::cubecl::dtype_to_storage_type, tensor::Shape};
use burn_cubecl::{CubeBackend, CubeRuntime, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim};

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime> Backend for CubeBackend<R> {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        // For simplicity, make sure each tensor is continuous.
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        // Get the matmul relevant shapes.
        let ndims = lhs.meta.num_dims();
        let num_rows = lhs.meta.shape()[ndims - 2];
        let num_cols = rhs.meta.shape()[ndims - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in shape_out.clone().into_iter().take(ndims - 2) {
            shape_out[i] = usize::max(lhs.meta.shape()[i], rhs.meta.shape()[i]);
            num_batches *= shape_out[i];
        }
        shape_out[ndims - 2] = num_rows;
        shape_out[ndims - 1] = num_cols;
        let shape_out = Shape::from(shape_out);

        // Create a buffer for the output tensor.
        let buffer = lhs.client.empty(shape_out.num_elements() * dtype.size());

        // Create the output tensor primitive.
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            dtype,
        );

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        fused_matmul_add_relu_kernel::launch(
            &output.client,
            cube_count,
            cube_dim,
            lhs.into_tensor_arg(),
            rhs.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(dtype),
        );

        // Return the output tensor.
        output
    }
}

impl<R: CubeRuntime> Backend for burn_fusion::Fusion<CubeBackend<R>> {
    fn fused_matmul_add_relu(
        _lhs: FloatTensor<Self>,
        _rhs: FloatTensor<Self>,
        _bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        todo!()
    }
}
