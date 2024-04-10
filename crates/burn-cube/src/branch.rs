use crate::{CubeContext, ExpandElement, UInt};
use burn_jit::gpu::{self, Variable};

pub fn range<S, E, F>(start: S, end: E, _unroll: bool, func: F)
where
    S: Into<UInt>,
    E: Into<UInt>,
    F: Fn(UInt),
{
    let start: UInt = start.into();
    let end: UInt = end.into();

    for i in start.val..end.val {
        func(UInt::new(i, 1))
    }
}

pub mod for_each {
    use burn_jit::gpu::{Branch, Elem, Item};

    use super::*;

    pub fn expand<F>(
        context: &mut CubeContext,
        start: ExpandElement,
        end: ExpandElement,
        unroll: bool,
        func: F,
    ) where
        F: Fn(&mut CubeContext, Variable),
    {
        if unroll {
            let start = match start.as_ref() {
                Variable::ConstantScalar(val, _) => *val as usize,
                _ => panic!("Only constant start can be unrolled."),
            };
            let end = match end.as_ref() {
                Variable::ConstantScalar(val, _) => *val as usize,
                _ => panic!("Only constant end can be unrolled."),
            };

            for i in start..end {
                func(context, i.into())
            }
        } else {
            let mut child = context.child();
            let index_ty = Item::Scalar(Elem::UInt);
            let i = child.scope.create_local_undeclared(index_ty);

            func(&mut child, i);

            context.scope.register(Branch::RangeLoop(gpu::RangeLoop {
                i,
                start: *start,
                end: *end,
                scope: child.scope,
            }));
        }
    }
}
