use alloc::rc::Rc;
use burn_jit::gpu::{Item, Variable};

/// Types used in a cube function must implement this trait
///
/// Variables whose values will be known at runtime must
/// have ExpandElement as associated type
/// Variables whose values will be known at compile time
/// must have the primitive type as associated type
///
/// Note: Cube functions should be written using RuntimeTypes,
/// so that the code generated uses the associated ExpandType.
/// This allows Cube code to not necessitate cloning, which is cumbersome
/// in algorithmic code. The necessary cloning will automatically appear in
/// the generated code.
pub trait RuntimeType {
    type ExpandType: Clone;
}

#[derive(new, Clone, Debug)]
/// Reference to a JIT variable
/// It's the expand element that is actually kept in the variable pool
pub struct ExpandElement {
    pub(crate) inner: Rc<Variable>,
}

impl ExpandElement {
    /// Returns the Item of the variable
    pub fn item(&self) -> Item {
        self.inner.item()
    }
}

impl From<u32> for ExpandElement {
    fn from(value: u32) -> Self {
        ExpandElement::new(Rc::new(Variable::from(value)))
    }
}

impl From<usize> for ExpandElement {
    fn from(value: usize) -> Self {
        ExpandElement::new(Rc::new(Variable::from(value)))
    }
}

impl From<bool> for ExpandElement {
    fn from(value: bool) -> Self {
        ExpandElement::new(Rc::new(Variable::from(value)))
    }
}

impl From<f32> for ExpandElement {
    fn from(value: f32) -> Self {
        ExpandElement::new(Rc::new(Variable::from(value)))
    }
}

impl core::ops::Deref for ExpandElement {
    type Target = Variable;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

#[derive(new, Clone, Copy)]
pub struct Float {
    pub val: f32,
    pub vectorization: u8,
}

#[derive(new, Clone, Copy)]
pub struct Int {
    pub val: u32,
    pub vectorization: u8,
}

#[derive(new, Clone, Copy)]
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

#[derive(new, Clone, Copy)]
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct Array<E> {
    pub vals: Vec<E>,
}

impl RuntimeType for Float {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<Float> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<Int> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<UInt> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Array<Bool> {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Int {
    type ExpandType = ExpandElement;
}

impl RuntimeType for UInt {
    type ExpandType = ExpandElement;
}

impl RuntimeType for Bool {
    type ExpandType = ExpandElement;
}

impl RuntimeType for bool {
    type ExpandType = bool;
}

impl RuntimeType for u32 {
    type ExpandType = u32;
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value, 1)
    }
}

impl RuntimeType for f32 {
    type ExpandType = f32;
}

impl From<f32> for Float {
    fn from(value: f32) -> Self {
        Float::new(value, 1)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32, 1)
    }
}

impl From<ExpandElement> for Variable {
    fn from(value: ExpandElement) -> Self {
        // Is it ok to do that?
        (*value.inner).clone()
    }
}
