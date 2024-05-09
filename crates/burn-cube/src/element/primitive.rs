use std::rc::Rc;

use burn_jit::gpu::Variable;

use crate::{ExpandElement, RuntimeType};

impl RuntimeType for bool {
    type ExpandType = bool;
}

impl RuntimeType for u32 {
    type ExpandType = u32;
}

impl RuntimeType for f32 {
    type ExpandType = f32;
}

impl RuntimeType for i32 {
    type ExpandType = i32;
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
