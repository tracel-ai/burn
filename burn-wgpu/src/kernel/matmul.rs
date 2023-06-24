use std::cmp::min;

use super::{build_info, DynamicKernelSettings, KernelSettings, SourceTemplate, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

const M: usize = 23;
const N: usize = 27;
const K: usize = 11;

const B_M: usize = 18; 
const B_N: usize = 23;
const B_K: usize = 17;
const T_M: usize = 30;


kernel_wgsl!(
    MatmulTiling1DRaw,
    "../template/lfd_matmul_blocktiling_1d.wgsl"
);

struct MatmulTiling1D;
// struct MatmulCaching;

impl StaticKernel for MatmulTiling1D {
    fn source_template() -> SourceTemplate {
        MatmulTiling1DRaw::source_template()
            .register("b_m", B_M.to_string())
            .register("b_n", B_N.to_string())
            .register("b_k", B_K.to_string())
            .register("bm_x_bk", (B_M * B_N).to_string())
            .register("bk_x_bn", (B_M * B_N).to_string())
            .register("t_m", T_M.to_string())
    }
}

pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_tiling_1d(lhs, rhs)
}

pub fn matmul_tiling_1d<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_save_device(&rhs);
    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];
    shape_out[D - 2] = num_rows;
    shape_out[D - 1] = num_cols;
    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    assert!(B_K <= min(B_M, B_N)); // otherwise there won't be enough threads to fill the shared memories in lhs and rhs

    const WORKGROUP_SIZE_X: usize = B_M;
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / B_M as f32) as u32;

    const WORKGROUP_SIZE_Y: usize = B_N;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / B_N as f32) as u32;

    // assert_eq!(WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, BLOCK_SIZE * BLOCK_K);

    let kernel =
        DynamicKernelSettings::<MatmulTiling1D, E, i32>::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1);
    let kernel = lhs.context.compile_dynamic(kernel);

    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

    println!("WorkGroup {:?}", workgroup);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestTensor;

    pub type ReferenceTensor<const D: usize> =
        burn_tensor::Tensor<burn_ndarray::NdArrayBackend<f32>, D>;

    #[test]
    pub fn test_tiling_1d() {
        same_as_reference(matmul_tiling_1d, [M, K], [K, N]);
    }

    fn same_as_reference<F, const D: usize, S>(func: F, shape_lhs: S, shape_rhs: S)
    where
        F: Fn(WgpuTensor<f32, D>, WgpuTensor<f32, D>) -> WgpuTensor<f32, D>,
        S: Into<Shape<D>>,
    {
        let x = ReferenceTensor::random(shape_lhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
        let y = ReferenceTensor::random(shape_rhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
        // let x = ReferenceTensor::ones(shape_lhs);
        // let y = ReferenceTensor::ones(shape_rhs);

        let x_wgpu = TestTensor::from_data(x.to_data());
        let y_wgpu = TestTensor::from_data(y.to_data());

        let z_reference = x.matmul(y);

        let z = func(x_wgpu.into_primitive(), y_wgpu.into_primitive());
        let z = TestTensor::from_primitive(z);

        println!("{z_reference}");
        println!("{z}");
        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }
}
