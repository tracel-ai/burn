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
    pub fn try_merge(&self, other: &Self) -> Option<Self> {
        if self.layout == other.layout {
            let mut globals = Vec::with_capacity(self.globals.len() + other.globals.len());
            globals.extend(&self.globals);
            globals.extend(&other.globals);

            let mut outs = Vec::with_capacity(self.outs.len() + other.outs.len());
            outs.extend(&self.outs);
            outs.extend(&other.outs);

            return Some(Self {
                globals,
                outs,
                layout: self.layout.clone(),
            });
        }

        None
    }

    pub fn expand(self, scope: &mut Scope) {
        let outputs = self.outs;
        let tensors = self.globals;
        let indexes = tensors
            .iter()
            .map(|_| scope.create_local(Elem::UInt))
            .collect::<Vec<_>>();

        OffsetGlobalWithLayoutAlgo {
            tensors: tensors.clone(),
            layout: self.layout,
            indexes: indexes.clone(),
            offset_ref: Variable::Id,
            end: Variable::Rank,
        }
        .expand(scope);

        for i in 0..outputs.len() {
            let tensor = tensors[i];
            let output = outputs[i];
            let index = indexes[i];

            gpu!(scope, output = tensor[index]);
        }
    }
}

#[derive(Debug, Clone)]
pub struct OffsetGlobalWithLayoutAlgo {
    pub tensors: Vec<Variable>,
    pub layout: Variable,
    pub indexes: Vec<Variable>,
    pub offset_ref: Variable,
    pub end: Variable,
}

impl OffsetGlobalWithLayoutAlgo {
    pub fn expand(self, scope: &mut Scope) {
        let layout = self.layout;
        let index_item_ty = Item::Scalar(Elem::UInt);
        let offset_ref = self.offset_ref;
        let zero: Variable = 0u32.into();
        let vectorization_factor: Variable = match self.tensors[0].item() {
            Item::Vec4(_) => 4u32,
            Item::Vec3(_) => 3u32,
            Item::Vec2(_) => 2u32,
            Item::Scalar(_) => 1u32,
        }
        .into();

        for index in self.indexes.iter() {
            gpu!(scope, index = zero);
        }
        gpu!(
            scope,
            range(zero, self.end).for_each(|i, scope| {
                let stride_layout = scope.create_local(index_item_ty);
                let ordso = scope.create_local(index_item_ty);

                gpu!(scope, stride_layout = stride(layout, i));
                gpu!(scope, ordso = offset_ref * vectorization_factor);
                gpu!(scope, ordso = ordso / stride_layout);

                for (tensor, index) in self.tensors.iter().zip(self.indexes.iter()) {
                    let stride = scope.create_local(index_item_ty);
                    let shape = scope.create_local(index_item_ty);
                    let tmp = scope.create_local(index_item_ty);

                    gpu!(scope, stride = stride(tensor, i));
                    gpu!(scope, shape = shape(tensor, i));

                    gpu!(scope, tmp = ordso % shape);
                    gpu!(scope, tmp = tmp * stride);
                    gpu!(scope, index = index + tmp);
                }
            })
        );

        for index in self.indexes {
            gpu!(scope, index = index / vectorization_factor);
        }
    }
}
