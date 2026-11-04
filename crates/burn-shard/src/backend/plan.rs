use std::collections::HashMap;

use burn_backend::{DType, ops::attention::AttentionShapes};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_std::Slice;
use smallvec::smallvec;

use super::classify::OpClass;
use crate::{
    AlongDimRule, AttentionRule, ElementwiseRule, LayoutRule, Linearity, MatmulRule, OpInputs,
    OpKind, OpOutputs, OpSharding, ReduceRule, ReducedDims, ReductionKind, ReshapeRule, Sharding,
};

/// Where each tensor of one op must be for the op to run on every member, where its outputs land,
/// and when the members run it.
#[derive(Debug)]
pub struct OpPlan {
    input_shardings: OpInputs<(TensorId, Sharding)>,
    output_shardings: OpOutputs<(TensorId, Sharding)>,
    execution: Execution,
}

/// When the members run one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    /// Now, each member on its own shards.
    Eager,
    /// When its output is first needed: split if that use splits it, else on member 0 and copied.
    Deferred,
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two shardings at once, gathers every
    /// input and runs whole on every member.
    pub fn new(
        op: &OperationIr,
        shardings: &HashMap<TensorId, Sharding>,
        num_members: usize,
    ) -> Self {
        TensorShardings {
            shardings,
            num_members,
        }
        .plan(OpClass::from(op))
        .unwrap_or_else(|| Self::gathered(op))
    }

    pub fn input_sharding(&self, id: &TensorId) -> Sharding {
        self.input_shardings
            .iter()
            .find(|(input, _)| input == id)
            .map(|(_, sharding)| *sharding)
            .expect("The plan has a sharding for every input of its op")
    }

    pub fn output_shardings(&self) -> &[(TensorId, Sharding)] {
        &self.output_shardings
    }

    pub fn execution(&self) -> Execution {
        self.execution
    }

    /// The plan a rule's `sharding` asks for, or `None` when one tensor, read as two inputs, is
    /// asked for two shardings, or when a quantized tensor would not be replicated: its shards
    /// cannot be sliced, concatenated or summed.
    fn from_rule(
        inputs: &[&TensorIr],
        out: &TensorIr,
        sharding: OpSharding,
        execution: Execution,
    ) -> Option<Self> {
        assert_eq!(
            inputs.len(),
            sharding.inputs.len(),
            "A rule gives one sharding per input"
        );
        if inputs
            .iter()
            .copied()
            .zip(sharding.inputs.iter().copied())
            .chain([(out, sharding.output)])
            .any(|(tensor, target)| {
                matches!(tensor.dtype, DType::QFloat(_)) && target != Sharding::Replicated
            })
        {
            return None;
        }
        let mut input_shardings = OpInputs::new();
        for (tensor, target) in inputs.iter().zip(sharding.inputs) {
            match input_shardings.iter().find(|(id, _)| *id == tensor.id) {
                Some((_, previous)) if *previous != target => return None,
                Some(_) => {}
                None => input_shardings.push((tensor.id, target)),
            }
        }
        Some(Self {
            input_shardings,
            output_shardings: smallvec![(out.id, sharding.output)],
            execution,
        })
    }

    fn gathered(op: &OperationIr) -> Self {
        let replicated = |tensor: &TensorIr| (tensor.id, Sharding::Replicated);
        Self {
            input_shardings: op.inputs().map(replicated).collect(),
            output_shardings: op.outputs().map(replicated).collect(),
            execution: Execution::Eager,
        }
    }
}

/// The shardings of a sharded compute device's tensors, which pick the rule each op runs by.
struct TensorShardings<'a> {
    shardings: &'a HashMap<TensorId, Sharding>,
    num_members: usize,
}

impl TensorShardings<'_> {
    fn plan(&self, class: OpClass) -> Option<OpPlan> {
        match class {
            OpClass::Kind { kind, inputs, out } => self.apply(kind, &inputs, out),
            OpClass::Gathered => None,
        }
    }

    /// The plan an op of `kind` runs by, or `None` when its inputs are not the ones the kind
    /// reads.
    fn apply(&self, kind: OpKind, inputs: &[&TensorIr], out: &TensorIr) -> Option<OpPlan> {
        match (kind, inputs) {
            (OpKind::Elementwise(linearity), _) => self.elementwise(linearity, inputs, out),
            (OpKind::AlongDim { dim }, [input]) => self.along_dim(dim, input, out),
            (
                OpKind::Reduction {
                    dims,
                    kind: reduction,
                },
                [input],
            ) => self.reduction(input, out, dims, reduction),
            (OpKind::Matmul, [lhs, rhs]) => self.matmul(lhs, rhs, out),
            (OpKind::Random, []) => self.random(out),
            (OpKind::Reshape, [input]) => self.reshape(input, out),
            (OpKind::SwapDims { dim1, dim2 }, [input]) => self.layout(
                input,
                out,
                LayoutRule::swap(self.sharding(input), dim1, dim2),
            ),
            (OpKind::Permute { axes }, [input]) => {
                self.layout(input, out, LayoutRule::permute(self.sharding(input), axes))
            }
            (OpKind::Slice { ranges }, [input]) => self.layout(
                input,
                out,
                LayoutRule::slice(self.sharding(input), ranges, &input.shape),
            ),
            (OpKind::SliceAssign { ranges }, [tensor, value]) => {
                self.slice_assign(tensor, value, ranges, out)
            }
            (OpKind::Expand, [input]) => self.layout(
                input,
                out,
                LayoutRule::expand(
                    self.sharding(input),
                    input.shape.num_dims(),
                    out.shape.num_dims(),
                ),
            ),
            (OpKind::Attention, _) => self.attention(inputs, out),
            _ => None,
        }
    }

    /// Broadcasting lines up dims of the same index, so an input with another number of dims has
    /// no rule, and neither has a linearity naming an input the op does not have.
    fn elementwise(
        &self,
        linearity: Linearity,
        inputs: &[&TensorIr],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let num_dims = out.shape.num_dims();
        if inputs.is_empty()
            || inputs
                .iter()
                .any(|input| input.shape.num_dims() != num_dims)
            || matches!(linearity, Linearity::LinearIn { input } if input >= inputs.len())
        {
            return None;
        }
        let shardings: OpInputs<_> = inputs.iter().map(|input| self.sharding(input)).collect();
        let shapes: OpInputs<_> = inputs.iter().map(|input| &input.shape).collect();
        let sharding = ElementwiseRule::new(linearity).sharding(&shardings, &shapes);
        OpPlan::from_rule(inputs, out, sharding, Execution::Eager)
    }

    fn along_dim(&self, dim: usize, input: &TensorIr, out: &TensorIr) -> Option<OpPlan> {
        let sharding = AlongDimRule::new(self.sharding(input), dim).sharding();
        OpPlan::from_rule(&[input], out, sharding, Execution::Eager)
    }

    fn reduction(
        &self,
        input: &TensorIr,
        out: &TensorIr,
        dims: ReducedDims,
        reduction: ReductionKind,
    ) -> Option<OpPlan> {
        let sharding = ReduceRule::new(self.sharding(input), dims, reduction).sharding();
        OpPlan::from_rule(&[input], out, sharding, Execution::Eager)
    }

    fn matmul(&self, lhs: &TensorIr, rhs: &TensorIr, out: &TensorIr) -> Option<OpPlan> {
        let sharding =
            MatmulRule::new(self.sharding(lhs), self.sharding(rhs), lhs.shape.num_dims())
                .sharding();
        OpPlan::from_rule(&[lhs, rhs], out, sharding, Execution::Eager)
    }

    /// The query, key and value come first, then the mask and bias the op has.
    fn attention(&self, inputs: &[&TensorIr], out: &TensorIr) -> Option<OpPlan> {
        let [query, key, value, ..] = inputs else {
            return None;
        };
        let rule = AttentionRule::new(
            [query, key, value].map(|input| self.sharding(input)),
            &AttentionShapes::new(&query.shape, &key.shape, &value.shape),
            self.num_members,
        );
        let shapes: OpInputs<_> = inputs.iter().map(|input| &input.shape).collect();
        OpPlan::from_rule(inputs, out, rule.sharding(&shapes), Execution::Eager)
    }

    fn reshape(&self, input: &TensorIr, out: &TensorIr) -> Option<OpPlan> {
        let sharding = ReshapeRule::new(
            self.sharding(input),
            &input.shape,
            &out.shape,
            self.num_members,
        )
        .sharding();
        OpPlan::from_rule(&[input], out, sharding, Execution::Eager)
    }

    fn layout(&self, input: &TensorIr, out: &TensorIr, rule: LayoutRule) -> Option<OpPlan> {
        OpPlan::from_rule(&[input], out, rule.sharding(1), Execution::Eager)
    }

    fn slice_assign(
        &self,
        tensor: &TensorIr,
        value: &TensorIr,
        ranges: &[Slice],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let rule = LayoutRule::slice_assign(
            self.sharding(tensor),
            self.sharding(value),
            ranges,
            &tensor.shape,
        );
        OpPlan::from_rule(&[tensor, value], out, rule.sharding(2), Execution::Eager)
    }

    fn random(&self, out: &TensorIr) -> Option<OpPlan> {
        let sharding = OpSharding {
            inputs: OpInputs::new(),
            output: Sharding::Replicated,
        };
        OpPlan::from_rule(&[], out, sharding, Execution::Deferred)
    }

    fn sharding(&self, tensor: &TensorIr) -> Sharding {
        self.shardings[&tensor.id]
    }
}
