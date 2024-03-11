use super::base::{matmul_tiling_2d_launch, B_K, B_M, B_N, WORKGROUP_SIZE};
use crate::{
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate, StaticKernelSource},
    tensor::JitTensor,
};
use crate::{kernel_wgsl, Runtime};
use std::marker::PhantomData;

kernel_wgsl!(
    MatmulTiling2Dvec4Raw,
    "../../template/matmul/blocktiling_2d/vec4.wgsl"
);

#[derive(new, Debug)]
struct MatmulTiling2Dvec4<E: JitElement> {
    _elem: PhantomData<E>,
}

impl<E: JitElement> DynamicKernelSource for MatmulTiling2Dvec4<E> {
    fn source(&self) -> SourceTemplate {
        MatmulTiling2Dvec4Raw::source()
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

pub fn matmul_tiling_2d_padded<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let kernel = MatmulTiling2Dvec4::<E>::new();
    matmul_tiling_2d_launch::<R, _, D, _>(lhs, rhs, out, kernel)
}
