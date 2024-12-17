use cubecl::prelude::Numeric;
use std::marker::PhantomData;

/// Matrix multiplication spec definiting each element types used in the computation as well as
/// how the arguments are passed to the kernel.
pub trait ConvSpec: Send + Sync + Clone + 'static {
    /// The plane size used by this kernel.
    const PLANE_DIM: u32;

    /// Element type of each input and output tensor of the kernel.
    type EG: Numeric;
    /// Element type of the intermediate representation of the inputs.
    type ES: Numeric;
    /// Element type of the intermediate representation of the output accumulator.
    type EA: Numeric;
}

/// Specification for a simple standard matmul using global tensor as inputs.
#[derive(Clone)]
pub struct SingleConvSpec<const PLANE_DIM: u32, EG: Numeric, ES: Numeric, EA: Numeric> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _ea: PhantomData<EA>,
}

impl<EG: Numeric, ES: Numeric, EA: Numeric, const PLANE_DIM: u32> ConvSpec
    for SingleConvSpec<PLANE_DIM, EG, ES, EA>
{
    const PLANE_DIM: u32 = PLANE_DIM;

    type EG = EG;
    type ES = ES;
    type EA = EA;
}
