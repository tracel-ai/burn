use burn_backend::{DType, ops::attention::AttentionShapes};
use burn_ir::{AttentionOpIr, CustomOpIr, OperationIr, TensorId, TensorIr};
use burn_std::{Slice, params_shape};

use super::{CustomOpPlan, classify::OpClass, custom_ops::CustomOpRuleMap, shardings::Shardings};
use crate::{
    AlongDimRule, AttentionRule, ElementwiseRule, GridSampleRule, IndexNdRule, IndexRule,
    LayoutRule, Linearity, MatmulRule, OpInputs, OpKind, OpOutputs, OpSharding, ReduceRule,
    ReducedDims, ReductionKind, ReshapeRule, Resharding, Sharding, ShardingError,
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
    /// Each member on its own shards.
    EveryMember,
    /// The first member alone, its output copied to every other. A random draw runs so: its values
    /// depend on the launch, which can differ between members.
    FirstMemberCopied,
    /// Once over every member's shards together, by a custom op's collective handler.
    Collective,
    /// Not at all: its outputs fail on every member, since its inputs cannot take the shardings its
    /// rule asks for.
    Refused(ShardingError),
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two shardings at once, gathers every
    /// input and runs whole on every member.
    pub fn new(
        op: &OperationIr,
        shardings: &Shardings,
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

    pub fn output_sharding(&self, id: &TensorId) -> Sharding {
        self.output_shardings
            .iter()
            .find(|(output, _)| output == id)
            .map(|(_, sharding)| *sharding)
            .expect("The plan has a sharding for every output of its op")
    }

    pub fn output_shardings(&self) -> &[(TensorId, Sharding)] {
        &self.output_shardings
    }

    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    /// The plan a rule's `sharding` asks for, or `None` when one tensor, read as two inputs, is
    /// asked for two shardings.
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
        inputs: impl IntoIterator<Item = (&'a TensorIr, Sharding)>,
        outputs: impl IntoIterator<Item = (&'a TensorIr, Sharding)>,
        execution: Execution,
    ) -> Option<Self> {
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

    /// This plan with `others` landing as its first output does, as an op's indices land with
    /// its values.
    fn with_outputs_alike(mut self, others: &[&TensorIr]) -> Self {
        if let Some(&(_, sharding)) = self.output_shardings.first() {
            let others = others.iter().map(|other| (other.id, sharding));
            self.output_shardings.extend(others);
        }
        self
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
            execution: Execution::EveryMember,
        }
    }
}

/// The shardings of a sharded compute device's tensors, which pick the rule each op runs by.
struct TensorShardings<'a> {
    shardings: &'a Shardings,
    rules: &'a CustomOpRuleMap,
    num_members: usize,
}

impl TensorShardings<'_> {
    fn plan(&self, class: OpClass) -> Option<OpPlan> {
        match class {
            OpClass::Kind {
                kind,
                inputs,
                outputs,
            } => self.apply(kind, &inputs, &outputs),
            OpClass::Custom(op) => self.custom(op),
            OpClass::CachedAttention(desc) => self.cached_attention(desc),
            OpClass::Gathered => None,
        }
    }

    /// The plan an op of `kind` runs by, or `None` when its inputs are not the ones the kind
    /// reads, or when it would move a quantized tensor to anything but whole or the split it
    /// already has, which cuts it through the host, or split a quantized output inside its
    /// blocks or packed words, or one whose per-tensor scale a member could recompute.
    fn apply(&self, kind: OpKind, inputs: &[&TensorIr], outputs: &[&TensorIr]) -> Option<OpPlan> {
        let (out, others) = outputs.split_first()?;
        self.rule_plan(kind, inputs, out)
            .map(|plan| plan.with_outputs_alike(others))
            .filter(|plan| self.keeps_quantized(kind, plan, inputs, outputs))
    }

    fn keeps_quantized(
        &self,
        kind: OpKind,
        plan: &OpPlan,
        inputs: &[&TensorIr],
        outputs: &[&TensorIr],
    ) -> bool {
        let quantized = |tensor: &TensorIr| matches!(tensor.dtype, DType::QFloat(_));
        let inputs_kept = inputs.iter().filter(|input| quantized(input)).all(|input| {
            let target = plan.input_sharding(&input.id);
            target == Sharding::Replicated || target == self.sharding(input)
        });
        let outputs_kept = outputs.iter().filter(|out| quantized(out)).all(|out| {
            let target = plan.output_sharding(&out.id);
            target == Sharding::Replicated
                || Self::keeps_scales(kind, out)
                    && Resharding::of_tensor(Sharding::Replicated, target, out, self.num_members)
                        .is_ok()
        });
        inputs_kept && outputs_kept
    }

    /// Whether each member's part of a quantized output holds the whole's scales. A quantize
    /// reads its scales. An op that only moves values may requantize them, which keeps every
    /// block's scale but not a per-tensor scale drawn from one member's part. Any other op may
    /// quantize its result by another scheme than its output's.
    fn keeps_scales(kind: OpKind, out: &TensorIr) -> bool {
        let block_scales_only =
            matches!(out.dtype, DType::QFloat(scheme) if scheme.tensor_scale().is_none());
        match kind {
            OpKind::Quantize => true,
            OpKind::Reshape
            | OpKind::SwapDims { .. }
            | OpKind::Permute { .. }
            | OpKind::Flip { .. }
            | OpKind::Index { .. }
            | OpKind::Select { .. }
            | OpKind::Slice { .. }
            | OpKind::Expand => block_scales_only,
            _ => false,
        }
    }

    fn rule_plan(&self, kind: OpKind, inputs: &[&TensorIr], out: &TensorIr) -> Option<OpPlan> {
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
            (OpKind::Random, []) => self.whole(out, Execution::FirstMemberCopied),
            (OpKind::Creation, []) => self.whole(out, Execution::EveryMember),
            (OpKind::Quantize, [input, params @ ..]) => self.quantize(input, params, out),
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
            (OpKind::Concat { dim }, [_, ..]) => {
                let shardings: OpInputs<_> =
                    inputs.iter().map(|input| self.sharding(input)).collect();
                let rule = LayoutRule::concat(&shardings, dim);
                OpPlan::from_rule(
                    inputs,
                    out,
                    rule.sharding(inputs.len()),
                    Execution::EveryMember,
                )
            }
            (OpKind::Select { dim }, [input, indices]) => self.select(dim, input, indices, out),
            (OpKind::SelectAssign { dim }, [tensor, indices, value]) => {
                self.select_assign(dim, tensor, indices, value, out)
            }
            (OpKind::Index { dim }, [_, _] | [_, _, _]) => self.index(dim, inputs, out),
            (OpKind::GatherNd, [data, indices]) => {
                let sharding = self.index_nd(data, indices)?.gather_sharding();
                OpPlan::from_rule(inputs, out, sharding, Execution::EveryMember)
            }
            (OpKind::ScatterNd, [data, indices, _]) => {
                let sharding = self.index_nd(data, indices)?.scatter_sharding();
                OpPlan::from_rule(inputs, out, sharding, Execution::EveryMember)
            }
            (OpKind::Flip { axes }, [input]) => {
                self.layout(input, out, LayoutRule::flip(self.sharding(input), axes))
            }
            (OpKind::Pad { padding }, [input]) => {
                let sharding = AlongDimRule::pad(self.sharding(input), padding).sharding();
                OpPlan::from_rule(inputs, out, sharding, Execution::EveryMember)
            }
            (OpKind::GridSample, [input, _]) => {
                let sharding = GridSampleRule::new(self.sharding(input)).sharding();
                OpPlan::from_rule(inputs, out, sharding, Execution::EveryMember)
            }
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
        OpPlan::from_rule(inputs, out, sharding, Execution::EveryMember)
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
        OpPlan::from_rule(&inputs, out, sharding, Execution::EveryMember)
    }

    /// A select keeps its input's split along any other dim, each member picking from its own
    /// part.
    fn select(
        &self,
        dim: usize,
        input: &TensorIr,
        indices: &TensorIr,
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let mut sharding = AlongDimRule::new(self.sharding(input), dim).sharding();
        sharding.inputs.push(Sharding::Replicated);
        OpPlan::from_rule(&[input, indices], out, sharding, Execution::EveryMember)
    }

    /// A select_assign keeps a split of its tensor or values along another dim, each member
    /// writing its own part at every index.
    fn select_assign(
        &self,
        dim: usize,
        tensor: &TensorIr,
        indices: &TensorIr,
        value: &TensorIr,
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let rule = IndexRule::new(
            &[self.sharding(tensor), self.sharding(value)],
            &[&tensor.shape, &value.shape],
            dim,
        );
        let mut sharding = rule.sharding(2);
        sharding.inputs.insert(1, Sharding::Replicated);
        OpPlan::from_rule(
            &[tensor, indices, value],
            out,
            sharding,
            Execution::EveryMember,
        )
    }

    /// A quantize keeps its input's split, each member quantizing its own blocks by its part of
    /// the block scales. A per-tensor scale, of one element, is whole on every member.
    fn quantize(&self, input: &TensorIr, params: &[&TensorIr], out: &TensorIr) -> Option<OpPlan> {
        let DType::QFloat(scheme) = out.dtype else {
            return None;
        };
        // Scales of another shape than the block grid are read in block order, which a split
        // cannot follow.
        let grid = params_shape(&input.shape, &scheme);
        let [scales, global @ ..] = params else {
            return None;
        };
        if scales.shape != grid || global.iter().any(|scale| scale.shape.num_elements() != 1) {
            return None;
        }
        let num_dims = input.shape.num_dims();
        let sharding = match self.sharding(input) {
            Sharding::PartialSum => Sharding::Replicated,
            sharding => sharding,
        };
        let params = params.iter().map(|param| {
            (
                *param,
                ElementwiseRule::input_sharding(sharding, &param.shape, num_dims),
            )
        });
        OpPlan::from_shardings(
            [(input, sharding)].into_iter().chain(params),
            [(out, sharding)],
            Execution::EveryMember,
        )
    }

    fn index(&self, dim: usize, inputs: &[&TensorIr], out: &TensorIr) -> Option<OpPlan> {
        let shardings: OpInputs<_> = inputs.iter().map(|input| self.sharding(input)).collect();
        let shapes: OpInputs<_> = inputs.iter().map(|input| &input.shape).collect();
        let rule = IndexRule::new(&shardings, &shapes, dim);
        OpPlan::from_rule(
            inputs,
            out,
            rule.sharding(inputs.len()),
            Execution::EveryMember,
        )
    }

    /// The rule of a gather_nd or scatter_nd, or `None` for indices without a dim to hold the
    /// tuples.
    fn index_nd(&self, data: &TensorIr, indices: &TensorIr) -> Option<IndexNdRule> {
        let batch_dims = indices.shape.num_dims().checked_sub(1)?;
        Some(IndexNdRule::new(
            self.sharding(data),
            indices.shape[batch_dims],
            batch_dims,
        ))
    }

    fn reduction(
        &self,
        input: &TensorIr,
        out: &TensorIr,
        dims: ReducedDims,
        reduction: ReductionKind,
    ) -> Option<OpPlan> {
        let sharding = ReduceRule::new(self.sharding(input), dims, reduction).sharding();
        OpPlan::from_rule(&[input], out, sharding, Execution::EveryMember)
    }

    fn matmul(&self, lhs: &TensorIr, rhs: &TensorIr, out: &TensorIr) -> Option<OpPlan> {
        let sharding =
            MatmulRule::new(self.sharding(lhs), self.sharding(rhs), lhs.shape.num_dims())
                .sharding();
        OpPlan::from_rule(&[lhs, rhs], out, sharding, Execution::EveryMember)
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
        OpPlan::from_rule(inputs, out, rule.sharding(&shapes), Execution::EveryMember)
    }

    /// An attention over a key/value cache runs on each member's own heads, its lengths and page
    /// table whole on every member, or gathers: a split by sequence would cut the pools and the
    /// page table, which are not laid out by sequence.
    fn cached_attention(&self, desc: &AttentionOpIr) -> Option<OpPlan> {
        let (query, key, value) = (&desc.query, &desc.key, &desc.value);
        // Both cache layouts keep the heads on dim 1, as the query does.
        let rule = AttentionRule::of_heads(
            [query, key, value].map(|input| self.sharding(input)),
            query.shape[1],
            key.shape[1],
            self.num_members,
        );
        if rule != AttentionRule::Heads {
            return None;
        }
        let attended: OpInputs<&TensorIr> = [query, key, value]
            .into_iter()
            .chain(desc.mask.as_ref())
            .chain(desc.attn_bias.as_ref())
            .collect();
        let shapes: OpInputs<_> = attended.iter().map(|input| &input.shape).collect();
        let heads = rule.sharding(&shapes);
        let whole = [desc.lengths.as_ref(), desc.pages.as_ref()]
            .into_iter()
            .flatten()
            .map(|input| (input, Sharding::Replicated));
        OpPlan::from_shardings(
            attended
                .iter()
                .copied()
                .zip(heads.inputs.iter().copied())
                .chain(whole),
            [(&desc.out, heads.output)],
            Execution::EveryMember,
        )
    }

    fn reshape(&self, input: &TensorIr, out: &TensorIr) -> Option<OpPlan> {
        let sharding = ReshapeRule::new(
            self.sharding(input),
            &input.shape,
            &out.shape,
            self.num_members,
        )
        .sharding();
        OpPlan::from_rule(&[input], out, sharding, Execution::EveryMember)
    }

    fn layout(&self, input: &TensorIr, out: &TensorIr, rule: LayoutRule) -> Option<OpPlan> {
        OpPlan::from_rule(&[input], out, rule.sharding(1), Execution::EveryMember)
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
        OpPlan::from_rule(
            &[tensor, value],
            out,
            rule.sharding(2),
            Execution::EveryMember,
        )
    }

    /// An op with no inputs makes its output whole on every member.
    fn whole(&self, out: &TensorIr, execution: Execution) -> Option<OpPlan> {
        let sharding = OpSharding {
            inputs: OpInputs::new(),
            output: Sharding::Replicated,
        };
        OpPlan::from_rule(&[], out, sharding, execution)
    }

    /// The plan its rule gives a custom op, or `None` to gather it: without a rule, or for a
    /// kind that does not plan it.
    fn custom(&self, op: &CustomOpIr) -> Option<OpPlan> {
        let rule = self.rules.get(&op.id)?;
        match rule.plan(op) {
            CustomOpPlan::Gathered => None,
            CustomOpPlan::Kind(kind) => self.custom_kind(kind, op),
            CustomOpPlan::Collective { inputs, outputs } => {
                Some(self.explicit(op, &inputs, &outputs, Execution::Collective))
            }
            CustomOpPlan::Sharded { inputs, outputs } => {
                Some(self.explicit(op, &inputs, &outputs, Execution::EveryMember))
            }
        }
    }

    /// A kind plans one output. A random op is drawn once and a slice cut to each member's ranges
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
            (kind, [out]) => self.apply(kind, &op.inputs.iter().collect::<Vec<_>>(), &[out]),
            _ => None,
        }
    }

    /// The plan of a custom op whose rule names every tensor's sharding, run as `execution`
    /// says: collective or by each member on its shards. Refused when a tensor cannot take its
    /// sharding: the rule reads the IR alone, so it cannot know what each input holds now.
    ///
    /// # Panics
    ///
    /// When the rule gives another number of shardings than the op has tensors, or two
    /// shardings for one tensor it reads twice.
    fn explicit(
        &self,
        op: &CustomOpIr,
        inputs: &[Sharding],
        outputs: &[Sharding],
        execution: Execution,
    ) -> OpPlan {
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
        if let Some(error) = self.explicit_refusal(op, inputs, outputs) {
            return OpPlan::refused(error);
        }
        OpPlan::from_shardings(
            op.inputs.iter().zip(inputs.iter().copied()),
            op.outputs.iter().zip(outputs.iter().copied()),
            execution,
        )
        .unwrap_or_else(|| {
            panic!(
                "The rule of `{}` gives two shardings to one tensor it reads twice",
                op.id
            )
        })
    }

    fn explicit_refusal(
        &self,
        op: &CustomOpIr,
        inputs: &[Sharding],
        outputs: &[Sharding],
    ) -> Option<ShardingError> {
        let input_refusal = op.inputs.iter().zip(inputs).find_map(|(input, &target)| {
            Resharding::of_tensor(self.sharding(input), target, input, self.num_members).err()
        });
        let output_refusal = || {
            op.outputs
                .iter()
                .zip(outputs)
                .find_map(|(out, &target)| match target.split_dim() {
                    Some(_) => {
                        Resharding::of_tensor(Sharding::Replicated, target, out, self.num_members)
                            .err()
                    }
                    None => None,
                })
        };
        input_refusal.or_else(output_refusal)
    }

    fn sharding(&self, tensor: &TensorIr) -> Sharding {
        self.shardings.get(&tensor.id)
    }
}
