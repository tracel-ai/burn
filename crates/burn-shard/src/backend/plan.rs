use std::collections::HashMap;

use burn_backend::{DType, ops::attention::AttentionShapes};
use burn_ir::{CustomOpIr, OperationIr, TensorId, TensorIr};
use burn_std::Slice;

use super::{CustomOpPlan, classify::OpClass, custom_ops::CustomOpRuleMap};
use crate::{
    AlongDimRule, AttentionRule, ElementwiseRule, LayoutRule, Linearity, MatmulRule, OpInputs,
    OpKind, OpOutputs, OpSharding, ReduceRule, ReducedDims, ReductionKind, ReshapeRule, Resharding,
    Sharding, ShardingError,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Execution {
    /// Now, each member on its own shards.
    Eager,
    /// When its output is first needed: split if that use splits it, else on member 0 and copied.
    Deferred,
    /// Now, once over every member's shards together, by a custom op's collective handler.
    Collective,
    /// Never: its outputs fail on every member, since its inputs cannot take the shardings its
    /// rule asks for.
    Refused(ShardingError),
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two shardings at once, gathers every
    /// input and runs whole on every member.
    pub fn new(
        op: &OperationIr,
        shardings: &HashMap<TensorId, Sharding>,
        rules: &CustomOpRuleMap,
        num_members: usize,
    ) -> Self {
        TensorShardings {
            shardings,
            rules,
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

    pub fn execution(&self) -> &Execution {
        &self.execution
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
        Self::from_shardings(
            inputs.iter().copied().zip(sharding.inputs.iter().copied()),
            [(out, sharding.output)],
            execution,
        )
    }

    /// The plan of each tensor's target sharding, or `None` as for
    /// [`from_rule`](Self::from_rule).
    fn from_shardings<'a>(
        inputs: impl IntoIterator<Item = (&'a TensorIr, Sharding)> + Clone,
        outputs: impl IntoIterator<Item = (&'a TensorIr, Sharding)> + Clone,
        execution: Execution,
    ) -> Option<Self> {
        if inputs
            .clone()
            .into_iter()
            .chain(outputs.clone())
            .any(|(tensor, target)| {
                matches!(tensor.dtype, DType::QFloat(_)) && target != Sharding::Replicated
            })
        {
            return None;
        }
        let mut input_shardings = OpInputs::new();
        for (tensor, target) in inputs {
            match input_shardings.iter().find(|(id, _)| *id == tensor.id) {
                Some((_, previous)) if *previous != target => return None,
                Some(_) => {}
                None => input_shardings.push((tensor.id, target)),
            }
        }
        Some(Self {
            input_shardings,
            output_shardings: outputs
                .into_iter()
                .map(|(tensor, target)| (tensor.id, target))
                .collect(),
            execution,
        })
    }

    /// The plan of an op that never runs, its outputs failing with `error`.
    fn refused(error: ShardingError) -> Self {
        Self {
            input_shardings: OpInputs::new(),
            output_shardings: OpOutputs::new(),
            execution: Execution::Refused(error),
        }
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
    rules: &'a CustomOpRuleMap,
    num_members: usize,
}

impl TensorShardings<'_> {
    fn plan(&self, class: OpClass) -> Option<OpPlan> {
        match class {
            OpClass::Kind { kind, inputs, out } => self.apply(kind, &inputs, out),
            OpClass::Custom(op) => self.custom(op),
            OpClass::Gathered => None,
        }
    }

    /// The plan an op of `kind` runs by, or `None` when its inputs are not the ones the kind
    /// reads.
    fn apply(&self, kind: OpKind, inputs: &[&TensorIr], out: &TensorIr) -> Option<OpPlan> {
        match (kind, inputs) {
            (OpKind::Elementwise(linearity), _) => self.elementwise(linearity, inputs, out),
            (OpKind::AlongDim { dim }, [input, broadcast @ ..]) => {
                self.along_dim(dim, input, broadcast, out)
            }
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

    /// An input with fewer dims lines them up with the output's last ones, and one with more, or a
    /// linearity naming an input the op does not have, has no rule.
    fn elementwise(
        &self,
        linearity: Linearity,
        inputs: &[&TensorIr],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let num_dims = out.shape.num_dims();
        if inputs.is_empty()
            || inputs.iter().any(|input| input.shape.num_dims() > num_dims)
            || matches!(linearity, Linearity::LinearIn { input } if input >= inputs.len())
        {
            return None;
        }
        let shardings: OpInputs<_> = inputs.iter().map(|input| self.sharding(input)).collect();
        let shapes: OpInputs<_> = inputs.iter().map(|input| &input.shape).collect();
        let sharding = ElementwiseRule::new(linearity).sharding(&shardings, &shapes, num_dims);
        OpPlan::from_rule(inputs, out, sharding, Execution::Eager)
    }

    /// The inputs after the first broadcast against it, like a norm's weight.
    fn along_dim(
        &self,
        dim: usize,
        input: &TensorIr,
        broadcast: &[&TensorIr],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let num_dims = input.shape.num_dims();
        if broadcast
            .iter()
            .any(|other| other.shape.num_dims() > num_dims)
        {
            return None;
        }
        let mut sharding = AlongDimRule::new(self.sharding(input), dim).sharding();
        sharding.inputs.extend(
            broadcast.iter().map(|other| {
                ElementwiseRule::input_sharding(sharding.output, &other.shape, num_dims)
            }),
        );
        let inputs: OpInputs<_> = [input]
            .into_iter()
            .chain(broadcast.iter().copied())
            .collect();
        OpPlan::from_rule(&inputs, out, sharding, Execution::Eager)
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

    /// The plan its rule gives a custom op, or `None` to gather it: without a rule, or for a
    /// kind that does not plan it.
    fn custom(&self, op: &CustomOpIr) -> Option<OpPlan> {
        let rule = self.rules.get(&op.id)?;
        match rule.plan(op) {
            CustomOpPlan::Gathered => None,
            CustomOpPlan::Kind(kind) => self.custom_kind(kind, op),
            CustomOpPlan::Collective { inputs, outputs } => {
                Some(self.collective(op, &inputs, &outputs))
            }
        }
    }

    /// A kind plans one output. A random op is deferred and a slice cut to each member's ranges
    /// only as their built-in IR, and a reshape or expand carries its target shape in its own
    /// scalars, which every member reads whole, so a custom op of those kinds gathers.
    fn custom_kind(&self, kind: OpKind, op: &CustomOpIr) -> Option<OpPlan> {
        match (kind, op.outputs.as_slice()) {
            (
                OpKind::Random
                | OpKind::Slice { .. }
                | OpKind::SliceAssign { .. }
                | OpKind::Reshape
                | OpKind::Expand,
                _,
            ) => None,
            (kind, [out]) => self.apply(kind, &op.inputs.iter().collect::<Vec<_>>(), out),
            _ => None,
        }
    }

    /// The plan of a collective custom op, refused when a tensor cannot take its sharding: the
    /// rule reads the IR alone, so it cannot know what each input holds now.
    ///
    /// # Panics
    ///
    /// When the rule gives another number of shardings than the op has tensors, or two
    /// shardings for one tensor it reads twice.
    fn collective(&self, op: &CustomOpIr, inputs: &[Sharding], outputs: &[Sharding]) -> OpPlan {
        assert!(
            inputs.len() == op.inputs.len() && outputs.len() == op.outputs.len(),
            "The rule of `{}` gives {} input and {} output shardings to an op of {} inputs and {} \
             outputs",
            op.id,
            inputs.len(),
            outputs.len(),
            op.inputs.len(),
            op.outputs.len(),
        );
        if let Some(error) = self.collective_refusal(op, inputs, outputs) {
            return OpPlan::refused(error);
        }
        OpPlan::from_shardings(
            op.inputs.iter().zip(inputs.iter().copied()),
            op.outputs.iter().zip(outputs.iter().copied()),
            Execution::Collective,
        )
        .unwrap_or_else(|| {
            panic!(
                "The rule of `{}` gives two shardings to one tensor it reads twice",
                op.id
            )
        })
    }

    fn collective_refusal(
        &self,
        op: &CustomOpIr,
        inputs: &[Sharding],
        outputs: &[Sharding],
    ) -> Option<ShardingError> {
        let quantized_split = |tensor: &TensorIr, target: Sharding| {
            matches!(tensor.dtype, DType::QFloat(_)) && target != Sharding::Replicated
        };
        let input_refusal = op.inputs.iter().zip(inputs).find_map(|(input, &target)| {
            if quantized_split(input, target) {
                return Some(ShardingError::Quantized);
            }
            Resharding::new(self.sharding(input), target, &input.shape, self.num_members).err()
        });
        let output_refusal = || {
            op.outputs.iter().zip(outputs).find_map(|(out, &target)| {
                if quantized_split(out, target) {
                    return Some(ShardingError::Quantized);
                }
                match target {
                    Sharding::Split { .. } => {
                        Resharding::new(Sharding::Replicated, target, &out.shape, self.num_members)
                            .err()
                    }
                    _ => None,
                }
            })
        };
        input_refusal.or_else(output_refusal)
    }

    fn sharding(&self, tensor: &TensorIr) -> Sharding {
        self.shardings[&tensor.id]
    }
}
