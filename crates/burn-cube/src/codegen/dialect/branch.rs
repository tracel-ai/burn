use super::{Elem, Item, Scope, Variable};
use serde::{Deserialize, Serialize};

/// All branching types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Branch {
    /// An if statement.
    If(If),
    /// An if else statement.
    IfElse(IfElse),
    /// A range loop.
    RangeLoop(RangeLoop),
    /// A loop.
    Loop(Loop),
    /// A return statement.
    Return,
    /// A break statement.
    Break,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct If {
    pub cond: Variable,
    pub scope: Scope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct IfElse {
    pub cond: Variable,
    pub scope_if: Scope,
    pub scope_else: Scope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct RangeLoop {
    pub i: Variable,
    pub start: Variable,
    pub end: Variable,
    pub scope: Scope,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct Loop {
    pub scope: Scope,
}

impl If {
    /// Registers an if statement to the given scope.
    pub fn register<F: Fn(&mut Scope)>(parent_scope: &mut Scope, cond: Variable, func: F) {
        let mut scope = parent_scope.child();

        func(&mut scope);

        let op = Self { cond, scope };
        parent_scope.register(Branch::If(op));
    }
}

impl IfElse {
    /// Registers an if else statement to the given scope.
    pub fn register<IF, ELSE>(
        parent_scope: &mut Scope,
        cond: Variable,
        func_if: IF,
        func_else: ELSE,
    ) where
        IF: Fn(&mut Scope),
        ELSE: Fn(&mut Scope),
    {
        let mut scope_if = parent_scope.child();
        let mut scope_else = parent_scope.child();

        func_if(&mut scope_if);
        func_else(&mut scope_else);

        parent_scope.register(Branch::IfElse(Self {
            cond,
            scope_if,
            scope_else,
        }));
    }
}

impl RangeLoop {
    /// Registers a range loop to the given scope.
    pub fn register<F: Fn(Variable, &mut Scope)>(
        parent_scope: &mut Scope,
        start: Variable,
        end: Variable,
        func: F,
    ) {
        let mut scope = parent_scope.child();
        let index_ty = Item::Scalar(Elem::UInt);
        let i = scope.create_local_undeclared(index_ty);

        func(i, &mut scope);

        parent_scope.register(Branch::RangeLoop(Self {
            i,
            start,
            end,
            scope,
        }));
    }
}

impl Loop {
    /// Registers a loop to the given scope.
    pub fn register<F: Fn(&mut Scope)>(parent_scope: &mut Scope, func: F) {
        let mut scope = parent_scope.child();

        func(&mut scope);

        let op = Self { scope };
        parent_scope.register(Branch::Loop(op));
    }
}

#[allow(missing_docs)]
pub struct UnrolledRangeLoop;

impl UnrolledRangeLoop {
    /// Registers an unrolled range loop to the given scope.
    pub fn register<F: Fn(Variable, &mut Scope)>(scope: &mut Scope, start: u32, end: u32, func: F) {
        for i in start..end {
            func(i.into(), scope);
        }
    }
}
