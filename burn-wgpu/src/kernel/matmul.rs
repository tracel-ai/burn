use std::cmp::min;

use super::{build_info, DynamicKernelSettings, KernelSettings, SourceTemplate, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

const bat_: usize = 2;
const bat: usize = 3;
const M: usize = 19;
const N: usize = 21;
const K: usize = 12;

const B_M: usize = 128;
const B_N: usize = 128;
const B_K: usize = 4;
const T_M: usize = 4;
const T_N: usize = 5;

// ne peut jamais y avoir plus de 1024 threads
//(B_M/T_M) * (B_N/T_N)

kernel_wgsl!(
    MatmulTiling2DRaw,
    "../template/lfd_matmul_blocktiling_2d.wgsl"
);

struct MatmulTiling2D;
// struct MatmulCaching;

impl StaticKernel for MatmulTiling2D {
    fn source_template() -> SourceTemplate {
        MatmulTiling2DRaw::source_template()
            .register("b_m", B_M.to_string())
            .register("b_n", B_N.to_string())
            .register("b_k", B_K.to_string())
            .register("bm_x_bn", (B_M * B_N).to_string())
            .register("bm_x_bk", (B_M * B_K).to_string())
            .register("bk_x_bn", (B_K * B_N).to_string())
            .register("t_m", T_M.to_string())
            .register("t_n", T_N.to_string())
            .register("tm_x_tn", (T_M * T_N).to_string())
    }
}

pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_tiling_2d(lhs, rhs)
}

pub fn matmul_tiling_2d<E: WgpuElement, const D: usize>(
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

    let n_threads_x: usize = f32::ceil(B_M as f32 / T_M as f32) as usize;
    let n_threads_y: usize = f32::ceil(B_N as f32 / T_N as f32) as usize;

    assert!(B_K <= min(B_M, B_N)); // otherwise not enough threads to fill B_K
    assert!(B_K * min(B_M, B_N) <= 8192); // otherwise uses too much memory

    let workgroup_size_x: usize = n_threads_x;
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / (n_threads_x * T_M) as f32) as u32;

    let workgroup_size_y: usize = n_threads_y;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / (n_threads_y * T_N) as f32) as u32;

    println!("{:?}", blocks_needed_in_x);
    println!("{:?}", blocks_needed_in_y);

    // assert_eq!(WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, BLOCK_SIZE * BLOCK_K);

    let kernel =
        DynamicKernelSettings::<MatmulTiling2D, E, i32>::new(workgroup_size_x, workgroup_size_y, 1);
    let kernel = lhs.context.compile_dynamic(kernel);

    let info = build_info(&[&lhs, &rhs, &output]);
    println!("{:?}", info);
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

// pub fn matmul_tiling_1d<E: WgpuElement, const D: usize>(
//     lhs: WgpuTensor<E, D>,
//     rhs: WgpuTensor<E, D>,
// ) -> WgpuTensor<E, D> {
//     lhs.assert_is_on_save_device(&rhs);
//     let mut shape_out = [0; D];
//     lhs.shape
//         .dims
//         .iter()
//         .zip(rhs.shape.dims.iter())
//         .enumerate()
//         .for_each(|(index, (dim_lhs, dim_rhs))| {
//             shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
//         });

//     let num_rows = lhs.shape.dims[D - 2];
//     let num_cols = rhs.shape.dims[D - 1];
//     shape_out[D - 2] = num_rows;
//     shape_out[D - 1] = num_cols;
//     let shape_out = Shape::new(shape_out);

//     let buffer = lhs
//         .context
//         .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
//     let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

//     const N_THREADS_X: usize = B_M;
//     const N_THREADS_Y: usize = B_N;

//     assert!(B_K <= min(N_THREADS_X, N_THREADS_Y)); // otherwise there won't be enough threads to fill the shared memories in lhs and rhs

//     const WORKGROUP_SIZE_X: usize = N_THREADS_X;
//     let blocks_needed_in_x = f32::ceil(num_rows as f32 / N_THREADS_X as f32) as u32;

//     const WORKGROUP_SIZE_Y: usize = N_THREADS_Y;
//     let blocks_needed_in_y = f32::ceil(num_cols as f32 / N_THREADS_Y as f32) as u32;

//     // assert_eq!(WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, BLOCK_SIZE * BLOCK_K);

//     let kernel =
//         DynamicKernelSettings::<MatmulTiling1D, E, i32>::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1);
//     let kernel = lhs.context.compile_dynamic(kernel);

//     let info = build_info(&[&lhs, &rhs, &output]);
//     let info_buffers = lhs
//         .context
//         .create_buffer_with_data(bytemuck::cast_slice(&info));

//     let mut num_iter = 1;
//     for i in 0..D - 2 {
//         num_iter *= output.shape.dims[i];
//     }

//     let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

//     println!("WorkGroup {:?}", workgroup);

//     lhs.context.execute(
//         workgroup,
//         kernel,
//         &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
//     );

//     output
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestTensor;

    pub type ReferenceTensor<const D: usize> =
        burn_tensor::Tensor<burn_ndarray::NdArrayBackend<f32>, D>;

    #[test]
    pub fn test_tiling_1d() {
        same_as_reference(matmul_tiling_2d, [M, K], [K, N]); //[bat_, bat, M, K], [bat_, bat, K, N]);
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
