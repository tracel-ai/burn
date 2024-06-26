use crate::{
    ir::{Operator, Variable, Vectorization},
    prelude::{init_expand, CubeContext, KernelBuilder, KernelLauncher},
    Runtime,
};
use alloc::rc::Rc;

use super::UInt;

/// Types used in a cube function must implement this trait
///
/// Variables whose values will be known at runtime must
/// have ExpandElement as associated type
/// Variables whose values will be known at compile time
/// must have the primitive type as associated type
///
/// Note: Cube functions should be written using CubeTypes,
/// so that the code generated uses the associated ExpandType.
/// This allows Cube code to not necessitate cloning, which is cumbersome
/// in algorithmic code. The necessary cloning will automatically appear in
/// the generated code.
pub trait CubeType {
    type ExpandType: Clone + Init;
}

pub trait Init {
    fn init(self, context: &mut CubeContext) -> Self;
}

/// Defines how a [launch argument](LaunchArg) can be expanded.
///
/// Normally this type should be implemented two times for an argument.
/// Once for the reference and the other for the mutable reference. Often time, the reference
/// should expand the argument as an input while the mutable reference should expand the argument
/// as an output.
pub trait LaunchArgExpand: CubeType {
    /// Register an input variable during compilation that fill the [KernelBuilder].
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> <Self as CubeType>::ExpandType;
}

/// Defines a type that can be used as argument to a kernel.
pub trait LaunchArg: CubeType {
    /// The runtime argument for the kernel.
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;
}

/// Defines the argument settings used to launch a kernel.
pub trait ArgSettings<R: Runtime>: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register(&self, launcher: &mut KernelLauncher<R>);
}

/// Reference to a JIT variable
#[derive(Clone, Debug)]
pub enum ExpandElement {
    /// Variable kept in the variable pool.
    Managed(Rc<Variable>),
    /// Variable not kept in the variable pool.
    Plain(Variable),
}

impl ExpandElement {
    pub fn can_mut(&self) -> bool {
        match self {
            ExpandElement::Managed(var) => {
                if let Variable::Local(_, _, _) = var.as_ref() {
                    Rc::strong_count(var) <= 2
                } else {
                    false
                }
            }
            ExpandElement::Plain(_) => false,
        }
    }
}

impl core::ops::Deref for ExpandElement {
    type Target = Variable;

    fn deref(&self) -> &Self::Target {
        match self {
            ExpandElement::Managed(var) => var.as_ref(),
            ExpandElement::Plain(var) => var,
        }
    }
}

impl From<ExpandElement> for Variable {
    fn from(value: ExpandElement) -> Self {
        match value {
            ExpandElement::Managed(var) => *var,
            ExpandElement::Plain(var) => var,
        }
    }
}

impl Init for ExpandElement {
    fn init(self, context: &mut CubeContext) -> Self {
        init_expand(context, self, Operator::Assign)
    }
}

macro_rules! impl_init_for {
    ($($t:ty),*) => {
        $(
            impl Init for $t {
                fn init(self, _context: &mut CubeContext) -> Self {
                    self
                }
            }
        )*
    };
}

// Add all types used within comptime
impl_init_for!(u32, bool, UInt);

impl<T> Init for Option<T> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}
