use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, Variable, Vectorization};
use crate::prelude::{KernelBuilder, KernelLauncher};
use crate::{
    frontend::{ArgSettings, Comptime},
    LaunchArg, Runtime,
};

#[derive(Clone, Copy, Debug)]
/// An unsigned int.
/// Preferred for indexing operations
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

impl CubeType for UInt {
    type ExpandType = ExpandElement;
}

impl CubeElem for UInt {
    fn as_elem() -> Elem {
        Elem::UInt
    }
}

impl LaunchArg for UInt {
    type RuntimeArg<'a, R: Runtime> = u32;

    fn compile_input(builder: &mut KernelBuilder, vectorization: Vectorization) -> ExpandElement {
        assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
        builder.scalar(Self::as_elem())
    }

    fn compile_output(builder: &mut KernelBuilder, vectorization: Vectorization) -> ExpandElement {
        assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
        builder.scalar(Self::as_elem())
    }
}

impl<R: Runtime> ArgSettings<R> for u32 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(*self);
    }
}

impl Numeric for UInt {}

impl UInt {
    pub const fn new(val: u32) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    pub fn new_expand(_context: &mut CubeContext, val: u32) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
        ExpandElement::Plain(new_var)
    }
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value)
    }
}

impl From<Comptime<u32>> for UInt {
    fn from(value: Comptime<u32>) -> Self {
        UInt::new(value.inner)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32)
    }
}

impl From<i32> for UInt {
    fn from(value: i32) -> Self {
        UInt::new(value as u32)
    }
}
