use std::marker::PhantomData;

use crate::{
    ir::{Operator, Variable, Vectorization},
    prelude::{init_expand, CubeContext, KernelBuilder, KernelLauncher},
    KernelSettings, Runtime,
};
use alloc::rc::Rc;

use super::{UInt, Vectorized};

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

/// Trait to be implemented by [cube types](CubeType) implementations.
pub trait Init: Sized {
    /// Initialize a type within a [context](CubeContext).
    ///
    /// You can return the same value when the variable is a non-mutable data structure or
    /// if the type can not be deeply cloned/copied.
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
    /// Register an output variable during compilation that fill the [KernelBuilder].
    fn expand_output(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> <Self as CubeType>::ExpandType {
        Self::expand(builder, vectorization)
    }
}

/// Defines a type that can be used as argument to a kernel.
pub trait LaunchArg: LaunchArgExpand + Send + Sync + 'static {
    /// The runtime argument for the kernel.
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;
}

impl LaunchArg for () {
    type RuntimeArg<'a, R: Runtime> = ();
}

impl<R: Runtime> ArgSettings<R> for () {
    fn register(&self, _launcher: &mut KernelLauncher<R>) {
        // nothing to do
    }
}

impl LaunchArgExpand for () {
    fn expand(
        _builder: &mut KernelBuilder,
        _vectorization: Vectorization,
    ) -> <Self as CubeType>::ExpandType {
    }
}

impl CubeType for () {
    type ExpandType = ();
}

impl Init for () {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

/// Defines the argument settings used to launch a kernel.
pub trait ArgSettings<R: Runtime>: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register(&self, launcher: &mut KernelLauncher<R>);
    /// Configure an input argument at the given position.
    fn configure_input(&self, _position: usize, settings: KernelSettings) -> KernelSettings {
        settings
    }
    /// Configure an output argument at the given position.
    fn configure_output(&self, _position: usize, settings: KernelSettings) -> KernelSettings {
        settings
    }
}

/// Reference to a JIT variable
#[derive(Clone, Debug)]
pub enum ExpandElement {
    /// Variable kept in the variable pool.
    Managed(Rc<Variable>),
    /// Variable not kept in the variable pool.
    Plain(Variable),
}

/// Expand type associated with a type.
#[derive(new)]
pub struct ExpandElementTyped<T> {
    pub(crate) expand: ExpandElement,
    pub(crate) _type: PhantomData<T>,
}

impl<T> Vectorized for ExpandElementTyped<T> {
    fn vectorization_factor(&self) -> UInt {
        self.expand.vectorization_factor()
    }

    fn vectorize(self, factor: UInt) -> Self {
        Self {
            expand: self.expand.vectorize(factor),
            _type: PhantomData,
        }
    }
}

impl<T> Clone for ExpandElementTyped<T> {
    fn clone(&self) -> Self {
        Self {
            expand: self.expand.clone(),
            _type: PhantomData,
        }
    }
}

impl<T> From<ExpandElement> for ExpandElementTyped<T> {
    fn from(expand: ExpandElement) -> Self {
        Self {
            expand,
            _type: PhantomData,
        }
    }
}

impl<T> From<ExpandElementTyped<T>> for ExpandElement {
    fn from(value: ExpandElementTyped<T>) -> Self {
        value.expand
    }
}

impl ExpandElement {
    pub fn can_mut(&self) -> bool {
        match self {
            ExpandElement::Managed(var) => {
                if let Variable::Local { .. } = var.as_ref() {
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
        if self.can_mut() {
            // Can reuse inplace :)
            return self;
        }

        let mut init = |elem: Self| init_expand(context, elem, Operator::Assign);

        match *self {
            Variable::GlobalScalar { .. } => init(self),
            Variable::LocalScalar { .. } => init(self),
            Variable::ConstantScalar { .. } => init(self),
            Variable::Local { .. } => init(self),
            // Constant should be initialized since the new variable can be mutated afterward.
            // And it is assumed those values are cloned.
            Variable::Rank
            | Variable::UnitPos
            | Variable::UnitPosX
            | Variable::UnitPosY
            | Variable::UnitPosZ
            | Variable::CubePos
            | Variable::CubePosX
            | Variable::CubePosY
            | Variable::CubePosZ
            | Variable::CubeDim
            | Variable::CubeDimX
            | Variable::CubeDimY
            | Variable::CubeDimZ
            | Variable::CubeCount
            | Variable::CubeCountX
            | Variable::CubeCountY
            | Variable::CubeCountZ
            | Variable::SubcubeDim
            | Variable::AbsolutePos
            | Variable::AbsolutePosX
            | Variable::AbsolutePosY
            | Variable::AbsolutePosZ => init(self),
            // Array types can't be copied, so we should simply return the same variable.
            Variable::SharedMemory { .. }
            | Variable::GlobalInputArray { .. }
            | Variable::GlobalOutputArray { .. }
            | Variable::LocalArray { .. }
            | Variable::Slice { .. }
            | Variable::Matrix { .. } => self,
        }
    }
}

macro_rules! impl_init_for {
    ($($t:ty),*) => {
        $(
            impl Init for $t {
                fn init(self, _context: &mut CubeContext) -> Self {
                    panic!("Shouln't be called, only for comptime.")
                }
            }

        )*
    };
}

// Add all types used within comptime
impl_init_for!(u32, bool, UInt);

impl<T: Init> Init for Option<T> {
    fn init(self, context: &mut CubeContext) -> Self {
        self.map(|o| Init::init(o, context))
    }
}

impl<T: CubeType> CubeType for Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: CubeType> CubeType for &mut Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: Init> Init for Vec<T> {
    fn init(self, context: &mut CubeContext) -> Self {
        self.into_iter().map(|e| e.init(context)).collect()
    }
}
