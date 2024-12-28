use super::args::FusedMatmulArgs;
use cubecl::{linalg::matmul::components::MatmulSpec, prelude::Numeric};
use std::marker::PhantomData;

/// Specification for a fused standard matmul.
#[derive(Clone)]
pub struct FusedMatmulSpec<const PLANE_DIM: u32, EG: Numeric, ES: Numeric, EA: Numeric> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _ea: PhantomData<EA>,
}

impl<EG: Numeric, ES: Numeric, EA: Numeric, const PLANE_DIM: u32> MatmulSpec
    for FusedMatmulSpec<PLANE_DIM, EG, ES, EA>
{
    type EG = EG;
    type ES = ES;
    type EA = EA;
    type Args = FusedMatmulArgs;
}
