use burn_tensor::Element;

use crate::{
    compute::DynamicKernel,
    element::JitElement,
    kernel::{into_contiguous, DynamicKernelSource, SourceTemplate, StaticKernelSource},
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

use crate::kernel_wgsl;

use super::base::{make_info_handle, make_workgroup, B_K, B_M, B_N, WORKGROUP_SIZE};

kernel_wgsl!(
    MatmulTiling2DUnpaddedRaw,
    "../../template/matmul/blocktiling_2d/unpadded.wgsl"
);

#[derive(new, Debug)]
struct MatmulTiling2DUnpadded<E: JitElement> {
    _elem: PhantomData<E>,
}

impl<E: JitElement> DynamicKernelSource for MatmulTiling2DUnpadded<E> {
    fn source(&self) -> SourceTemplate {
        MatmulTiling2DUnpaddedRaw::source()
            .register("b_m", B_M.to_string())
            .register("b_n", B_N.to_string())
            .register("b_k", B_K.to_string())
            .register("bm_x_bk_4", (B_M * B_K / 4).to_string())
            .register("bk_x_bn_4", (B_K * B_N / 4).to_string())
            .register("workgroup_size_x", WORKGROUP_SIZE.to_string())
            .register("workgroup_size_y", WORKGROUP_SIZE.to_string())
            .register("workgroup_size_z", "1".to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        std::format!("{:?}", self)
    }
}

/// Matrix multiplication using tiling 2d algorithm with
/// vec4 primitive on both lhs and rhs, with no padding needed
pub fn matmul_tiling_2d<R: Runtime, E: JitElement + Element, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let lhs = match lhs.batch_swapped_with_row_col() {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match rhs.batch_swapped_with_row_col() {
        true => into_contiguous(rhs),
        false => rhs,
    };

    let workgroup = make_workgroup(&out.shape);
    let info_handle = make_info_handle(&lhs, &rhs, &out);

    lhs.client.execute(
        Box::new(DynamicKernel::new(
            MatmulTiling2DUnpadded::<E>::new(),
            workgroup,
        )),
        &[&lhs.handle, &rhs.handle, &out.handle, &info_handle],
    );

    out
}
