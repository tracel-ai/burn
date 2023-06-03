use super::KernelSettings;
use crate::{context::WorkGroup, element::WGPUElement, kernel_wgsl, tensor::WGPUTensor};
use burn_tensor::Shape;
use num_traits::ToPrimitive;
use std::sync::Arc;

kernel_wgsl!(MatmulRaw, "../template/matmul.wgsl");

pub fn matmul<E: WGPUElement, const D: usize>(
    lhs: WGPUTensor<E, D>,
    rhs: WGPUTensor<E, D>,
) -> WGPUTensor<E, D> {
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
    let mut num_iter = 1;

    for i in 0..D - 2 {
        num_iter *= shape_out.dims[i];
    }

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WGPUTensor::new(lhs.context.clone(), shape_out, Arc::new(buffer));
    let kernel = lhs
        .context
        .compile::<KernelSettings<MatmulRaw, E, i32, 1, 16, 16>>();
    let mut info: Vec<u32> = vec![D.to_u32().unwrap()];

    lhs.strides
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    rhs.strides
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    lhs.shape
        .dims
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    rhs.shape
        .dims
        .into_iter()
        .for_each(|v| info.push(v.to_u32().unwrap()));
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let workgroup = WorkGroup::new(
        num_iter as u32,
        f32::ceil((lhs.shape.dims[D - 2] as f32) / 16.) as u32,
        f32::ceil((rhs.shape.dims[D - 1] as f32) / 16.) as u32,
    );

    println!(
        "AAAAAAAA {:?} % {:?} = {:?}",
        rhs.strides[0],
        rhs.shape.dims[0],
        rhs.shape.dims[0] % rhs.strides[0] * rhs.strides[0]
    );
    println!("WorkGroup {:?}", workgroup);

    lhs.context.execute(
        &workgroup,
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
