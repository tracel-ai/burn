use crate::frontend::{CubeElem, CubeType, ExpandElement};
use crate::ir::{Elem, Operator};
use crate::prelude::{init_expand, CubeContext};

use super::Init;

impl CubeType for bool {
    type ExpandType = ExpandElement;
}

impl CubeElem for bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}

// impl Init for bool {
//     fn init(self, context: &mut CubeContext) -> Self {
//         init_expand(context, self, Operator::Assign)
//     }
// }
