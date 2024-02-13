use super::{
    Elem, Item, Loop, Metadata, Operation, Operator, RangeLoop, Scope, UnaryOperator, Variable,
};
use crate::codegen::dialect::gpu::BinaryOperator;

pub struct ReadGlobalWithLayoutAlgo {
    pub operations_begin: Vec<Operation>,
    pub inner_loop: Loop,
    pub operations_end: Vec<Operation>,
}

pub struct ReadGlobalWithLayoutOperator {
    pub variable: Variable,
    pub layout: Variable,
}

pub struct AlgoGeneration {
    pub num_local_variables: u16,
    pub operations: Vec<Operation>,
}

pub fn generate_read_global_with_layout(
    prefix: String,
    num_local_variables: u16,
    algo: ReadGlobalWithLayoutOperator,
) -> AlgoGeneration {
    let mut scope = Scope {
        prefix,
        num_local_variables,
        operations: Vec::new(),
    };
    let index_type = Item::Scalar(Elem::UInt);
    let index_local = scope.create_local(index_type);
    let start = Variable::Constant(0.0, index_type);

    scope.register(Operator::AssignLocal(UnaryOperator {
        input: Variable::Constant(0.0, index_type),
        out: index_local.clone(),
    }));

    let offset = match algo.variable.item() {
        Item::Vec4(_) => 4.0,
        Item::Vec3(_) => 3.0,
        Item::Vec2(_) => 2.0,
        Item::Scalar(_) => 1.0,
    };
    let offset = Variable::Constant(offset, index_type);

    RangeLoop::new(&mut scope, start, Variable::Rank, |i, scope| {
        let stride = scope.create_local(index_type);
        let stride_layout = scope.create_local(index_type);
        let shape = scope.create_local(index_type);
        let numerator = scope.create_local(index_type);
        let denominator = scope.create_local(index_type);

        scope.register(Metadata::Stride {
            dim: i.clone(),
            var: algo.variable.clone(),
            out: stride.clone(),
        });
        scope.register(Metadata::Stride {
            dim: i.clone(),
            var: algo.layout.clone(),
            out: stride_layout.clone(),
        });
        scope.register(Metadata::Shape {
            dim: i.clone(),
            var: algo.variable.clone(),
            out: shape.clone(),
        });

        scope.register(Operator::Mul(BinaryOperator {
            lhs: Variable::Id,
            rhs: offset.clone(),
            out: numerator.clone(),
        }));
        scope.register(Operator::Modulo(BinaryOperator {
            lhs: stride_layout,
            rhs: shape,
            out: denominator.clone(),
        }));
        scope.register(Operator::Mul(BinaryOperator {
            lhs: denominator.clone(),
            rhs: stride,
            out: denominator.clone(),
        }));
        scope.register(Operator::Div(BinaryOperator {
            lhs: numerator.clone(),
            rhs: denominator.clone(),
            out: numerator.clone(),
        }));
        // +=
        scope.register(Operator::Add(BinaryOperator {
            lhs: index_local.clone(),
            rhs: numerator.clone(),
            out: index_local.clone(),
        }));
    });

    AlgoGeneration {
        num_local_variables: scope.num_local_variables,
        operations: scope.operations,
    }
}
