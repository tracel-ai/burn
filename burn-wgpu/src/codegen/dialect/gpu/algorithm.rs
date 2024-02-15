use super::{
    Elem, Item, Metadata, Operator, RangeLoop, ReadGlobalAlgo, ReadGlobalWithLayoutAlgo, Scope,
    UnaryOperator, Variable,
};
use crate::codegen::dialect::gpu::BinaryOperator;

impl ReadGlobalAlgo {
    pub fn expand(self, scope: &mut Scope) {
        scope.register(Operator::Index(BinaryOperator {
            lhs: self.global,
            rhs: Variable::Id,
            out: self.out,
        }));
    }
}

impl ReadGlobalWithLayoutAlgo {
    pub fn expand(self, scope: &mut Scope) {
        let index_item_ty = Item::Scalar(Elem::UInt);
        let index_local = scope.create_local(index_item_ty);
        let start = Variable::ConstantScalar(0.0, Elem::UInt);

        scope.register(Operator::AssignLocal(UnaryOperator {
            input: Variable::ConstantScalar(0.0, Elem::UInt),
            out: index_local,
        }));

        let offset = match self.global.item() {
            Item::Vec4(_) => 4.0,
            Item::Vec3(_) => 3.0,
            Item::Vec2(_) => 2.0,
            Item::Scalar(_) => 1.0,
        };
        let offset = Variable::ConstantScalar(offset, Elem::UInt);

        RangeLoop::register(scope, start, Variable::Rank, |i, scope| {
            let stride = scope.create_local(index_item_ty);
            let stride_layout = scope.create_local(index_item_ty);
            let shape = scope.create_local(index_item_ty);
            let tmp = scope.create_local(index_item_ty);

            scope.register(Metadata::Stride {
                dim: *i,
                var: self.global,
                out: stride,
            });
            scope.register(Metadata::Stride {
                dim: *i,
                var: self.layout,
                out: stride_layout,
            });
            scope.register(Metadata::Shape {
                dim: *i,
                var: self.global,
                out: shape,
            });

            scope.register(Operator::Mul(BinaryOperator {
                lhs: Variable::Id,
                rhs: offset,
                out: tmp,
            }));
            scope.register(Operator::Div(BinaryOperator {
                lhs: tmp,
                rhs: stride_layout,
                out: tmp,
            }));
            scope.register(Operator::Modulo(BinaryOperator {
                lhs: tmp,
                rhs: shape,
                out: tmp,
            }));
            scope.register(Operator::Mul(BinaryOperator {
                lhs: tmp,
                rhs: stride,
                out: tmp,
            }));
            scope.register(Operator::Add(BinaryOperator {
                lhs: index_local,
                rhs: tmp,
                out: index_local,
            }));
        });

        let tmp = scope.create_local(index_item_ty);
        scope.register(Operator::Div(BinaryOperator {
            lhs: index_local,
            rhs: offset,
            out: tmp,
        }));
        scope.register(Operator::Index(BinaryOperator {
            lhs: self.global,
            rhs: tmp,
            out: self.out,
        }));
    }
}
