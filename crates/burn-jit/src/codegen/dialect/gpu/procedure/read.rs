use super::super::{gpu, Elem, Item, Operator, Scope, Variable};
use crate::codegen::dialect::gpu::{BinaryOperator, Vectorization};
use serde::{Deserialize, Serialize};

/// Read a global array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadGlobal {
    /// The array to be read.
    pub global: Variable,
    /// The output variable to write the result.
    pub out: Variable,
    /// The reference position index.
    pub position: Variable,
}

/// Read a global array with the given layout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadGlobalWithLayout {
    /// The array to be read.
    pub globals: Vec<Variable>,
    /// The output variable to write the result.
    pub outs: Vec<Variable>,
    /// The layout to be used.
    pub layout: Variable,
    /// The reference position index.
    pub position: Variable,
}

impl ReadGlobal {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        scope.register(Operator::Index(BinaryOperator {
            lhs: self.global,
            rhs: self.position,
            out: self.out,
        }));
    }
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            global: self.global.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
            position: self.position,
        }
    }
}

impl ReadGlobalWithLayout {
    /// Try to merge two reads together reducing branching.
    pub fn try_merge(&self, other: &Self) -> Option<Self> {
        // Can only merge two reads when they share the same reference layout.
        if self.layout != other.layout {
            return None;
        }

        if self.position != other.position {
            return None;
        }

        let mut globals = Vec::with_capacity(self.globals.len() + other.globals.len());
        globals.extend(&self.globals);
        globals.extend(&other.globals);

        let mut outs = Vec::with_capacity(self.outs.len() + other.outs.len());
        outs.extend(&self.outs);
        outs.extend(&other.outs);

        Some(Self {
            globals,
            outs,
            layout: self.layout,
            position: self.position,
        })
    }

    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let outputs = self.outs;
        let tensors = self.globals;
        let indexes = tensors
            .iter()
            .map(|_| scope.create_local(Elem::UInt))
            .collect::<Vec<_>>();

        IndexOffsetGlobalWithLayout {
            tensors: tensors.clone(),
            layout: self.layout,
            indexes: indexes.clone(),
            position: self.position,
            dim_start: 0u32.into(),
            dim_end: Variable::Rank,
        }
        .expand(scope);

        for i in 0..outputs.len() {
            let tensor = tensors[i];
            let output = outputs[i];
            let index = indexes[i];

            gpu!(scope, output = tensor[index]);
        }
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            globals: self
                .globals
                .iter()
                .map(|g| g.vectorize(vectorization))
                .collect(),
            layout: self.layout.vectorize(vectorization),
            outs: self
                .outs
                .iter()
                .map(|o| o.vectorize(vectorization))
                .collect(),
            position: self.position,
        }
    }
}

/// Calculate the index offset for all tensor variables provided compatible with the given layout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct IndexOffsetGlobalWithLayout {
    /// Tensor [variables](Variable), same length as [indexes](Self::indexes).
    pub tensors: Vec<Variable>,
    /// Offsets that are going to be written to.
    pub indexes: Vec<Variable>,
    /// Reference layout.
    pub layout: Variable,
    /// Position index that corresponds to the reference layout.
    ///
    /// All other indexes will be made to be compatible with this one.
    pub position: Variable,
    pub dim_start: Variable,
    pub dim_end: Variable,
}

impl IndexOffsetGlobalWithLayout {
    #[allow(missing_docs)]
    pub fn expand(self, scope: &mut Scope) {
        let layout = self.layout;
        let index_item_ty = Item::Scalar(Elem::UInt);
        let offset_ref = self.position;
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
            range(self.dim_start, self.dim_end).for_each(|i, scope| {
                let stride_layout = scope.create_local(index_item_ty);
                let ogwl = scope.create_local(index_item_ty);

                gpu!(scope, stride_layout = stride(layout, i));
                gpu!(scope, ogwl = offset_ref * vectorization_factor);
                gpu!(scope, ogwl = ogwl / stride_layout);

                for (tensor, index) in self.tensors.iter().zip(self.indexes.iter()) {
                    let stride = scope.create_local(index_item_ty);
                    let shape = scope.create_local(index_item_ty);
                    let tmp = scope.create_local(index_item_ty);

                    gpu!(scope, stride = stride(tensor, i));
                    gpu!(scope, shape = shape(tensor, i));

                    gpu!(scope, tmp = ogwl % shape);
                    gpu!(scope, tmp = tmp * stride);
                    gpu!(scope, index = index + tmp);
                }
            })
        );

        for index in self.indexes {
            gpu!(scope, index = index / vectorization_factor);
        }
    }

    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            tensors: self
                .tensors
                .iter()
                .map(|t| t.vectorize(vectorization))
                .collect(),
            indexes: self
                .indexes
                .iter()
                .map(|t| t.vectorize(vectorization))
                .collect(),
            layout: self.layout.vectorize(vectorization),
            position: self.position.vectorize(vectorization),
            dim_start: self.dim_start,
            dim_end: self.dim_end,
        }
    }
}
