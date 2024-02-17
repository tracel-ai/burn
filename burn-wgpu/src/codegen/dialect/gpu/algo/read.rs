use super::super::{
    gpu, Elem, Item, Metadata, Operator, ReadGlobalAlgo, ReadGlobalWithLayoutAlgo, Scope, Variable,
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
        let out = self.out;
        let tensor = self.global;
        let index_local = scope.create_local(Elem::UInt);

        OffsetGlobalWithLayoutAlgo {
            global: tensor,
            layout: self.layout,
            out: index_local,
            offset_ref: Variable::Id,
            end: Variable::Rank,
        }
        .expand(scope);

        gpu!(scope, out = tensor[index_local]);
    }
}

#[derive(Debug, Clone)]
pub struct OffsetGlobalWithLayoutAlgo {
    pub global: Variable,
    pub layout: Variable,
    pub out: Variable,
    pub offset_ref: Variable,
    pub end: Variable,
}

impl OffsetGlobalWithLayoutAlgo {
    pub fn expand(self, scope: &mut Scope) {
        let tensor = self.global;
        let layout = self.layout;
        let index_item_ty = Item::Scalar(Elem::UInt);
        let output = self.out;
        let offset_ref = self.offset_ref;
        let zero: Variable = 0u32.into();
        let offset: Variable = match self.global.item() {
            Item::Vec4(_) => 4u32,
            Item::Vec3(_) => 3u32,
            Item::Vec2(_) => 2u32,
            Item::Scalar(_) => 1u32,
        }
        .into();

        gpu!(scope, output = zero);
        gpu!(
            scope,
            range(zero, self.end).for_each(|i, scope| {
                let stride = scope.create_local(index_item_ty);
                let stride_layout = scope.create_local(index_item_ty);
                let shape = scope.create_local(index_item_ty);
                let tmp = scope.create_local(index_item_ty);

                gpu!(scope, stride = stride(tensor, i));
                gpu!(scope, shape = shape(tensor, i));
                gpu!(scope, stride_layout = stride(layout, i));

                gpu!(scope, tmp = offset_ref * offset);
                gpu!(scope, tmp = tmp / stride_layout);
                gpu!(scope, tmp = tmp % shape);
                gpu!(scope, tmp = tmp * stride);
                gpu!(scope, output = output + tmp);
            })
        );

        gpu!(scope, output = output / offset);
    }
}
