use std::{ops::Deref, rc::Rc};

use crate::dialect::{Branch, Elem, If, IfElse, Item, Loop, RangeLoop, Variable};
use crate::language::{CubeContext, ExpandElement, UInt};
use crate::CubeType;

pub fn range<S, E>(start: S, end: E, _unroll: Comptime<bool>) -> impl Iterator<Item = UInt>
where
    S: Into<UInt>,
    E: Into<UInt>,
{
    let start: UInt = start.into();
    let end: UInt = end.into();

    (start.val..end.val).into_iter().map(UInt::new)
}

pub fn range_expand<F>(
    context: &mut CubeContext,
    start: ExpandElement,
    end: ExpandElement,
    unroll: bool,
    mut func: F,
) where
    F: FnMut(&mut CubeContext, ExpandElement),
{
    if unroll {
        let start = match start.deref() {
            Variable::ConstantScalar(val, _) => *val as usize,
            _ => panic!("Only constant start can be unrolled."),
        };
        let end = match end.deref() {
            Variable::ConstantScalar(val, _) => *val as usize,
            _ => panic!("Only constant end can be unrolled."),
        };

        for i in start..end {
            func(context, i.into())
        }
    } else {
        let mut child = context.child();
        let index_ty = Item::new(Elem::UInt);
        let i = child.scope.borrow_mut().create_local_undeclared(index_ty);
        let i = ExpandElement::new(Rc::new(i));

        func(&mut child, i.clone());

        context.register(Branch::RangeLoop(RangeLoop {
            i: *i,
            start: *start,
            end: *end,
            scope: child.into_scope(),
        }));
    }
}

pub fn if_expand<IF>(context: &mut CubeContext, cond: ExpandElement, mut block: IF)
where
    IF: FnMut(&mut CubeContext),
{
    let mut child = context.child();

    block(&mut child);

    context.register(Branch::If(If {
        cond: *cond,
        scope: child.into_scope(),
    }));
}

pub fn if_else_expand<IF, EL>(
    context: &mut CubeContext,
    cond: ExpandElement,
    mut then_block: IF,
    mut else_block: EL,
) where
    IF: FnMut(&mut CubeContext),
    EL: FnMut(&mut CubeContext),
{
    let mut then_child = context.child();
    then_block(&mut then_child);

    let mut else_child = context.child();
    else_block(&mut else_child);

    context.register(Branch::IfElse(IfElse {
        cond: *cond,
        scope_if: then_child.into_scope(),
        scope_else: else_child.into_scope(),
    }));
}

pub fn break_expand(context: &mut CubeContext) {
    context.register(Branch::Break);
}

pub fn loop_expand<FB>(context: &mut CubeContext, mut block: FB)
where
    FB: FnMut(&mut CubeContext),
{
    let mut inside_loop = context.child();

    block(&mut inside_loop);
    context.register(Branch::Loop(Loop {
        scope: inside_loop.into_scope(),
    }));
}

pub fn while_loop_expand<FC, FB>(context: &mut CubeContext, mut cond_fn: FC, mut block: FB)
where
    FC: FnMut(&mut CubeContext) -> ExpandElement,
    FB: FnMut(&mut CubeContext),
{
    let mut inside_loop = context.child();

    let cond: ExpandElement = cond_fn(&mut inside_loop);
    if_expand(&mut inside_loop, cond, break_expand);

    block(&mut inside_loop);
    context.register(Branch::Loop(Loop {
        scope: inside_loop.into_scope(),
    }));
}

#[derive(Clone, Copy)]
pub struct Comptime<T> {
    t: T,
}

impl<T> Comptime<T> {
    pub fn new(t: T) -> Self {
        Self { t }
    }

    pub fn get(comptime: Self) -> T {
        comptime.t
    }
}

impl<T: CubeType + Into<T::ExpandType>> Comptime<Option<T>> {
    pub fn is_some(comptime: Self) -> Comptime<bool> {
        Comptime::new(comptime.t.is_some())
    }
    pub fn value_or<F>(comptime: Self, mut alt: F) -> T
    where
        F: FnMut() -> T,
    {
        match comptime.t {
            Some(t) => t,
            None => alt(),
        }
    }

    pub fn value_or_expand<F>(
        context: &mut CubeContext,
        t: Option<T>,
        mut alt: F,
    ) -> <T as CubeType>::ExpandType
    where
        F: FnMut(&mut CubeContext) -> T::ExpandType,
    {
        match t {
            Some(t) => t.into(),
            None => alt(context),
        }
    }
}

impl<T: Clone> CubeType for Comptime<T> {
    type ExpandType = T;
}
