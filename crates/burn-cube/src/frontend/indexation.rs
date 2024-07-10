use super::{Comptime, ExpandElement, UInt};
use crate::ir::{Elem, Variable};

pub trait Index {
    fn value(self) -> Variable;
}

impl Index for Comptime<u32> {
    fn value(self) -> Variable {
        Variable::ConstantScalar {
            value: self.inner as f64,
            elem: Elem::UInt,
        }
    }
}

impl Index for Comptime<i32> {
    fn value(self) -> Variable {
        Variable::ConstantScalar {
            value: self.inner as f64,
            elem: Elem::UInt,
        }
    }
}

impl Index for i32 {
    fn value(self) -> Variable {
        Variable::ConstantScalar {
            value: self as f64,
            elem: Elem::UInt,
        }
    }
}

impl Index for u32 {
    fn value(self) -> Variable {
        Variable::ConstantScalar {
            value: self as f64,
            elem: Elem::UInt,
        }
    }
}

impl Index for UInt {
    fn value(self) -> Variable {
        Variable::ConstantScalar {
            value: self.val as f64,
            elem: Elem::UInt,
        }
    }
}

impl Index for ExpandElement {
    fn value(self) -> Variable {
        *self
    }
}
