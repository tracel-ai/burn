use super::args::FusedMatmulArgs;
use cubecl::linalg::matmul::components::{MatmulPrecision, MatmulSpec};
use std::marker::PhantomData;

/// Specification for a fused standard matmul.
#[derive(Clone)]
pub struct FusedMatmulSpec<EG: MatmulPrecision> {
    _eg: PhantomData<EG>,
}

impl<EG: MatmulPrecision> MatmulSpec for FusedMatmulSpec<EG> {
    type Precision = EG;
    type Args = FusedMatmulArgs;
}
