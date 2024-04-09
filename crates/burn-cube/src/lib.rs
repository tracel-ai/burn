extern crate alloc;
pub use burn_cube_macros::cube;

use alloc::sync::Arc;
use burn_jit::gpu::{self, Item, Scope, Variable};
use std::collections::HashMap;

pub struct CodegenContext<'a> {
    pub scope: &'a mut Scope,
    pub pool: HashMap<Item, Arc<Variable>>,
}

impl<'a> CodegenContext<'a> {
    pub fn crate_float(&mut self, item: Item) -> FloatVariable {
        let var = self.create_local(item);
        FloatVariable { var }
    }
    pub fn create_local(&mut self, item: Item) -> Arc<Variable> {
        for variable in self.pool.get(&item).iter() {
            if Arc::strong_count(variable) == 1 {
                return Arc::clone(variable);
            }
        }

        let new = Arc::new(self.scope.create_local(item));
        self.pool.insert(item, new.clone());

        new
    }
}

#[derive(Copy, Clone)]
pub struct Float {
    pub kind: u32,
    pub val: f32,
    pub vectorization: u8,
}

impl core::ops::Add for Float {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        panic!("Only used for types");
    }
}

#[derive(Clone)]
pub struct FloatVariable {
    var: Arc<Variable>,
}

impl CubeVariable for Float {
    type Variable = FloatVariable;
}

pub trait CubeVariable {
    type Variable: Clone;
}

pub fn float_add_expand(
    context: &mut CodegenContext<'_>,
    lhs: FloatVariable,
    rhs: FloatVariable,
) -> FloatVariable {
    let item = lhs.var.item();
    let out = context.create_local(item);
    let out = FloatVariable { var: out };

    let op = gpu::Operator::Add(gpu::BinaryOperator {
        lhs: *lhs.var,
        rhs: *rhs.var,
        out: *out.var,
    });

    context.scope.register(op);

    out
}

pub fn float_assign_expand(
    context: &mut CodegenContext<'_>,
    input: FloatVariable,
) -> FloatVariable {
    let item = input.var.item();
    let out = context.create_local(item);
    let out = FloatVariable { var: out };

    let op = gpu::Operator::Assign(gpu::UnaryOperator {
        input: *input.var,
        out: *out.var,
    });

    context.scope.register(op);

    out
}
