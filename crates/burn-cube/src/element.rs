use alloc::rc::Rc;
use burn_jit::gpu::{Item, Variable};

pub trait RuntimeType {
    type ExpandType: Clone;
}

#[derive(new, Clone)]
pub struct ExpandElement {
    pub(crate) inner: Rc<Variable>,
}

impl ExpandElement {
    pub fn item(&self) -> Item {
        self.inner.item()
    }
}

impl From<u32> for ExpandElement {
    fn from(value: u32) -> Self {
        ExpandElement::new(Rc::new(Variable::from(value)))
    }
}

impl core::ops::Deref for ExpandElement {
    type Target = Variable;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

#[derive(new, Clone)]
pub struct Float {
    pub val: f32,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct Int {
    pub val: u32,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

#[derive(new, Clone)]
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

impl RuntimeType for Float {
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

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value, 1)
    }
}
