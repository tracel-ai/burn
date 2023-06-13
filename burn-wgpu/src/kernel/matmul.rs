use super::{build_info, DynamicKernelSettings, KernelSettings, StaticKernelGenerator};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

const BLOCK_M: usize = 32;
const BLOCK_N: usize = 32;
const BLOCK_K: usize = 4;
const BLOCK_MK: usize = BLOCK_M * BLOCK_K;
const BLOCK_KN: usize = BLOCK_K * BLOCK_N;
const TILE_M: usize = 4;

kernel_wgsl!(MatmulNaiveRaw, "../template/matmul_naive.wgsl");
kernel_wgsl!(
    MatmulCoalescingRaw,
    "../template/matmul_mem_coalescing.wgsl"
);
kernel_wgsl!(MatmulCachingRaw, "../template/matmul_caching.wgsl");
kernel_wgsl!(MatmulTiling1DRaw, "../template/matmul_blocktiling_1d.wgsl");

struct MatmulTiling1D;
struct MatmulCoalescing;
struct MatmulCaching;

impl StaticKernelGenerator for MatmulTiling1D {
    type Source = String;

    fn generate() -> Self::Source {
        let source = MatmulTiling1DRaw::generate();

        let source = source.replace("{{BLOCK_M}}", &BLOCK_M.to_string());
        let source = source.replace("{{BLOCK_N}}", &BLOCK_N.to_string());
        let source = source.replace("{{BLOCK_K}}", &BLOCK_K.to_string());
        let source = source.replace("{{BLOCK_MK}}", &BLOCK_MK.to_string());
        let source = source.replace("{{BLOCK_KN}}", &BLOCK_KN.to_string());
        let source = source.replace("{{TILE_M}}", &TILE_M.to_string());

        source
    }
}

impl StaticKernelGenerator for MatmulCoalescing {
    type Source = String;

    fn generate() -> Self::Source {
        let source = MatmulCoalescingRaw::generate();

        let source = source.replace("BLOCK_SIZE_2X", &BLOCK_M.to_string());
        let source = source.replace("BLOCK_SIZE", &BLOCK_M.to_string());

        source
    }
}

impl StaticKernelGenerator for MatmulCaching {
    type Source = String;

    fn generate() -> Self::Source {
        let source = MatmulCoalescingRaw::generate();

        let source = source.replace("BLOCK_SIZE_2X", &BLOCK_M.to_string());
        let source = source.replace("BLOCK_SIZE", &BLOCK_M.to_string());

        source
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

    let kernel = DynamicKernelSettings::<MatmulTiling1D, E, i32>::new(BLOCK_M, BLOCK_M, 1);
    let kernel = lhs.context.compile_dynamic(kernel);

    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup_x = f32::ceil(num_rows as f32 / (BLOCK_M * TILE_M) as f32) as u32;
    let workgroup_y = f32::ceil(num_cols as f32 / (BLOCK_N * TILE_M) as f32) as u32;
    let workgroup = WorkGroup::new(workgroup_x, workgroup_y, num_iter as u32);

    println!("{:?}", workgroup);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}

pub fn matmul_coalescing<E: WgpuElement, const D: usize>(
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

    shape_out[D - 2] = lhs.shape.dims[D - 2];
    shape_out[D - 1] = rhs.shape.dims[D - 1];
    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);
    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<MatmulCoalescing, E, i32, BLOCK_M, BLOCK_M, 1>>();

    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup_x = f32::ceil(num_rows as f32 / BLOCK_M as f32) as u32;
    let workgroup_y = f32::ceil(num_cols as f32 / BLOCK_M as f32) as u32;
    let workgroup = WorkGroup::new(workgroup_x, workgroup_y, num_iter as u32);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}

pub fn matmul_caching<E: WgpuElement, const D: usize>(
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

    shape_out[D - 2] = lhs.shape.dims[D - 2];
    shape_out[D - 1] = rhs.shape.dims[D - 1];
    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);
    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<MatmulCaching, E, i32, BLOCK_M, BLOCK_M, 1>>();

    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup_x = f32::ceil(num_rows as f32 / BLOCK_M as f32) as u32;
    let workgroup_y = f32::ceil(num_cols as f32 / BLOCK_M as f32) as u32;
    let workgroup = WorkGroup::new(workgroup_x, workgroup_y, num_iter as u32);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}

pub fn matmul_naive<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_save_device(&rhs);
    const WORKGROUP_SIZE_X: usize = 16;
    const WORKGROUP_SIZE_Y: usize = 16;

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

    let kernel = lhs.context.compile_static::<KernelSettings<
        MatmulNaiveRaw,
        E,
        i32,
        WORKGROUP_SIZE_X,
        WORKGROUP_SIZE_Y,
        1,
    >>();

    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup_x = f32::ceil(num_rows as f32 / WORKGROUP_SIZE_X as f32) as u32;
    let workgroup_y = f32::ceil(num_cols as f32 / WORKGROUP_SIZE_Y as f32) as u32;
    let workgroup = WorkGroup::new(workgroup_x, workgroup_y, num_iter as u32);

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
    pub fn test_naive() {
        same_as_reference(matmul_naive, [4, 25, 13], [4, 13, 77]);
    }

    #[test]
    pub fn test_coalesing() {
        same_as_reference(matmul_coalescing, [4, 25, 13], [4, 13, 77]);
    }

    #[test]
    pub fn test_caching() {
        same_as_reference(matmul_caching, [4, 25, 13], [4, 13, 77]);
    }

    #[test]
    pub fn test_tiling_1d() {
        same_as_reference(matmul_tiling_1d, [16, 16], [16, 16]);
    }

    fn same_as_reference<F, const D: usize, S>(func: F, shape_lhs: S, shape_rhs: S)
    where
        F: Fn(WgpuTensor<f32, D>, WgpuTensor<f32, D>) -> WgpuTensor<f32, D>,
        S: Into<Shape<D>>,
    {
        let x = ReferenceTensor::random(shape_lhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
        let y = ReferenceTensor::random(shape_rhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));

        let x_wgpu = TestTensor::from_data(x.to_data());
        let y_wgpu = TestTensor::from_data(y.to_data());

        let z_reference = x.matmul(y);

        let z = func(x_wgpu.into_primitive(), y_wgpu.into_primitive());
        let z = TestTensor::from_primitive(z);

        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }
}
