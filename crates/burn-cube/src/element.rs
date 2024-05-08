use std::marker::PhantomData;

use alloc::rc::Rc;
use burn_jit::gpu::{Elem, Item, Variable};

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

impl From<i32> for ExpandElement {
    fn from(value: i32) -> Self {
        ExpandElement::new(Rc::new(Variable::from(value)))
    }
}

impl core::ops::Deref for ExpandElement {
    type Target = Variable;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

// Why _ suffixes? Just to avoid clashing with JIT float kind types
// TODO refactor
pub trait FloatKind_: Clone + Copy {
    fn to_elem() -> Elem;
}
#[derive(Clone, Copy)]
pub struct F32_;
#[derive(Clone, Copy)]
pub struct BF16_;
#[derive(Clone, Copy)]
pub struct F32_;
#[derive(Clone, Copy)]
pub struct F64_;
impl FloatKind_ for F32_ {
    fn to_elem() -> Elem {
        Elem::Float(burn_jit::gpu::FloatKind::F32)
    }
}
impl FloatKind_ for BF16_ {
    fn to_elem() -> Elem {
        Elem::Float(burn_jit::gpu::FloatKind::BF16)
    }
}
impl FloatKind_ for F32_ {
    fn to_elem() -> Elem {
        Elem::Float(burn_jit::gpu::FloatKind::F32)
    }
}
impl FloatKind_ for F64_ {
    fn to_elem() -> Elem {
        Elem::Float(burn_jit::gpu::FloatKind::F64)
    }
}

#[derive(Clone, Copy)]
pub struct Float<F: FloatKind_> {
    pub val: f32,
    pub vectorization: u8,
    pub _type: PhantomData<F>,
}

impl<F: FloatKind_> Float<F> {
    pub fn new(val: f32, vectorization: u8) -> Self {
        Self {
            val,
            vectorization,
            _type: PhantomData,
        }
    }
}

pub type Float_ = Float<F32_>;

#[derive(new, Clone, Copy)]
pub struct Int {
    pub val: i32,
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

impl<F: FloatKind_> RuntimeType for Float<F> {
    type ExpandType = ExpandElement;
}

impl<F: FloatKind_> RuntimeType for Array<Float<F>> {
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

impl<F: FloatKind_> From<f32> for Float<F> {
    fn from(value: f32) -> Self {
        Float::new(value, 1)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32, 1)
    }
}

impl RuntimeType for i32 {
    type ExpandType = i32;
}

impl From<i32> for Int {
    fn from(value: i32) -> Self {
        Int::new(value, 1)
    }
}

impl From<ExpandElement> for Variable {
    fn from(value: ExpandElement) -> Self {
        // Is it ok to do that?
        (*value.inner).clone()
    }
}
