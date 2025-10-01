use super::args::FusedMatmulArgs;
use cubecl::matmul::components::{MatmulPrecision, MatmulSpec};
use std::marker::PhantomData;

/// Specification for a fused standard matmul.
#[derive(Clone)]
pub struct FusedMatmulSpec<MP: MatmulPrecision> {
    _phantom: PhantomData<MP>,
}

impl<MP: MatmulPrecision> MatmulSpec for FusedMatmulSpec<MP> {
    type Precision = MP;
    type Args = FusedMatmulArgs;
}
