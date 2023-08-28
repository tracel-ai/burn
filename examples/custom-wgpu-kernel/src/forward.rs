use super::CustomBackend;
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

kernel_wgsl!(FusedMatmulAddReluRaw, "./kernel.wgsl");

#[derive(new, Debug)]
struct FusedMatmulAddRelu<E: FloatElement> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _elem: PhantomData<E>,
}

impl<E: FloatElement> DynamicKernel for FusedMatmulAddRelu<E> {
    fn source_template(self) -> SourceTemplate {
        FusedMatmulAddReluRaw::source_template()
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        std::format!("{:?}", self)
    }
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> CustomBackend for WgpuBackend<G, F, I> {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: WgpuTensor<F, D>,
        rhs: WgpuTensor<F, D>,
        bias: WgpuTensor<F, D>,
    ) -> WgpuTensor<F, D> {
        let workgroup_size_x = 16;
        let workgroup_size_y = 16;

        lhs.assert_is_on_same_device(&rhs);
        lhs.assert_is_on_same_device(&bias);

        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);
        let bias = into_contiguous(bias);

        let shape_out = shape_out(&lhs, &rhs);
        let num_rows = lhs.shape.dims[D - 2];
        let num_cols = rhs.shape.dims[D - 1];

        let buffer = lhs
            .context
            .create_buffer(shape_out.num_elements() * core::mem::size_of::<F>());
        let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

        // set number of workgroups
        let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
        let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;

        let kernel = lhs.context.compile_dynamic(FusedMatmulAddRelu::<F>::new(
            workgroup_size_x,
            workgroup_size_y,
        ));

        let info = build_info(&[&lhs, &rhs, &output]);

        let info_buffers = lhs
            .context
            .create_buffer_with_data(bytemuck::cast_slice(&info));

        let mut num_iter = 1;
        for i in 0..D - 2 {
            num_iter *= output.shape.dims[i];
        }

        let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

        lhs.context.execute(
            workgroup,
            kernel,
            &[
                &lhs.buffer,
                &rhs.buffer,
                &bias.buffer,
                &output.buffer,
                &info_buffers,
            ],
        );

        output
    }
}

pub(crate) fn shape_out<E: FloatElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
) -> Shape<D> {
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
    Shape::new(shape_out)
}
