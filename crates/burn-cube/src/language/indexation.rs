use crate::{
    dialect::{Elem, Variable},
    Comptime, ExpandElement, UInt,
};

pub trait Index {
    fn value(self) -> Variable;
}

impl Index for Comptime<u32> {
    fn value(self) -> Variable {
        Variable::ConstantScalar(self.inner as f64, Elem::UInt)
    }
}

impl Index for Comptime<i32> {
    fn value(self) -> Variable {
        Variable::ConstantScalar(self.inner as f64, Elem::UInt)
    }
}

impl Index for i32 {
    fn value(self) -> Variable {
        Variable::ConstantScalar(self as f64, Elem::UInt)
    }
}

impl Index for u32 {
    fn value(self) -> Variable {
        Variable::ConstantScalar(self as f64, Elem::UInt)
    }
}

impl Index for UInt {
    fn value(self) -> Variable {
        Variable::ConstantScalar(self.val as f64, Elem::UInt)
    }
}

impl Index for ExpandElement {
    fn value(self) -> Variable {
        *self
    }
}
