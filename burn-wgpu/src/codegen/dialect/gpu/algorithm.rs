use super::{
    Elem, Item, Metadata, Operator, RangeLoop, ReadGlobalWithLayoutAlgo, Scope, UnaryOperator,
    Variable,
};
use crate::codegen::dialect::gpu::{BinaryOperator, Loop};

pub fn generate_read_global_with_layout(scope: &mut Scope, algo: ReadGlobalWithLayoutAlgo) {
    assert!(
        scope.operations.is_empty(),
        "Scope must have empty operation"
    );
    println!("Generate global layout {:?}", algo);

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

    let op = RangeLoop::new(scope, start, Variable::Rank, |i, scope| {
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

    scope.register(Loop::Range(op));
    // scope.register(Operator::AssignLocal(UnaryOperator {
    //     input: todo!(), // TODO: Indexation with the input.
    //     out: algo.variable,
    // }));

    // {local} = {elem}({global}[index_{local} /  {offset}u]);
}
