use cubecl::{cube, prelude::*};

/// Declare a custom kernel that gets compiled to `wgpu`/`CUDA`
#[cube(launch)]
pub fn fused_matmul_add_relu_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let row = ABSOLUTE_POS_X;
    let col = ABSOLUTE_POS_Y;
    let batch = ABSOLUTE_POS_Z;

    let n_rows = output.shape(output.rank() - 2);
    let n_cols = output.shape(output.rank() - 1);
    let dim_k = rhs.shape(rhs.rank() - 1);

    if row >= n_rows || col >= n_cols {
        terminate!();
    }

    let offset_output = batch * n_rows * n_cols;
    let mut offset_lhs = 0;
    let mut offset_rhs = 0;

    let batch_dims = output.rank() - 2;
    for dim in 0..batch_dims {
        offset_lhs += offset_output / output.stride(dim) % lhs.shape(dim) * lhs.stride(dim);
        offset_rhs += offset_output / output.stride(dim) % rhs.shape(dim) * rhs.stride(dim);
    }

    let mut sum = F::new(0.0);
    for k in 0..dim_k {
        let lhs_index = row * dim_k + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let out_index = row * n_cols + col;
    let index = offset_output + out_index;

    output[index] = F::max(sum + bias[index], F::new(0.0));
}
