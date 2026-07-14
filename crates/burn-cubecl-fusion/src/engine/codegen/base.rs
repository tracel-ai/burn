use cubecl::{define_scalar, define_size, prelude::Vector};

define_scalar!(pub DynElem);
define_size!(pub DynSize);
pub type DynVector = Vector<DynElem, DynSize>;
