use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, Item, Variable, Vectorization};
use crate::prelude::{index_assign, KernelBuilder, KernelLauncher};
use crate::{
    frontend::{ArgSettings, Comptime},
    LaunchArg, Runtime,
};

use super::Vectorized;

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

    pub fn vectorized(val: u32, vectorization: UInt) -> Self {
        if vectorization.val == 1 {
            Self::new(val)
        } else {
            Self {
                val,
                vectorization: vectorization.val as u8,
            }
        }
    }

    pub fn vectorized_expand(
        context: &mut CubeContext,
        val: u32,
        vectorization: UInt,
    ) -> <Self as CubeType>::ExpandType {
        if vectorization.val == 1 {
            Self::new_expand(context, val)
        } else {
            let mut new_var =
                context.create_local(Item::vectorized(Self::as_elem(), vectorization.val as u8));
            for (i, element) in vec![val; vectorization.val as usize].iter().enumerate() {
                new_var = index_assign::expand(context, new_var, i, *element);
            }

            new_var
        }
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

impl Vectorized for UInt {
    fn vectorization_factor(&self) -> UInt {
        UInt {
            val: self.vectorization as u32,
            vectorization: 1,
        }
    }

    fn vectorize(mut self, factor: UInt) -> Self {
        self.vectorization = factor.vectorization;
        self
    }
}
