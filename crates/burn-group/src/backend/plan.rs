use std::collections::HashMap;

use burn_backend::DType;
use burn_ir::{
    ActivationOperationIr, BaseOperationIr, BinaryOpIr, FloatOperationIr, IntOperationIr,
    LinearOpIr, MatmulOpIr, ModuleOperationIr, NumericOperationIr, OperationIr, ScalarOpIr,
    TensorId, TensorIr, UnaryOpIr,
};

use super::placements::TensorPlacements;
use crate::{
    GroupPlacement, LinearRule, Linearity, MatmulRule, OpPlacement, ReduceRule, Reduction,
};

/// Where each tensor of one op must be for the op to run on every member, where its outputs land,
/// and what each member runs.
#[derive(Debug)]
pub struct OpPlan {
    inputs: HashMap<TensorId, GroupPlacement>,
    outputs: HashMap<TensorId, GroupPlacement>,
    execution: Execution,
}

/// What each member runs, when it is not simply the op on its own shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    EveryMember,
    /// Member 0 runs the op and the others take a copy, so random values agree.
    FirstMemberCopied,
    /// Only member 0 adds the bias: the output is a partial sum, which must count it once.
    BiasOnFirstMember,
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two placements at once, gathers every
    /// input and runs whole on every member.
    pub fn new(op: &OperationIr, placements: &TensorPlacements) -> Self {
        Analysis { placements }
            .plan(op)
            .unwrap_or_else(|| Self::gathered(op))
    }

    pub fn input(&self, id: &TensorId) -> GroupPlacement {
        self.inputs[id]
    }

    pub fn outputs(&self) -> &HashMap<TensorId, GroupPlacement> {
        &self.outputs
    }

    pub fn execution(&self) -> Execution {
        self.execution
    }

    fn gathered(op: &OperationIr) -> Self {
        let replicated = |tensor: &TensorIr| (tensor.id, GroupPlacement::Replicated);
        Self {
            inputs: op.inputs().map(replicated).collect(),
            outputs: op.outputs().map(replicated).collect(),
            execution: Execution::EveryMember,
        }
    }
}

struct Analysis<'a> {
    placements: &'a TensorPlacements,
}

impl Analysis<'_> {
    fn plan(&self, op: &OperationIr) -> Option<OpPlan> {
        match op {
            OperationIr::NumericFloat(dtype, op) | OperationIr::NumericInt(dtype, op) => {
                self.numeric(*dtype, op)
            }
            OperationIr::Float(_, FloatOperationIr::Matmul(desc))
            | OperationIr::Int(IntOperationIr::Matmul(desc)) => self.matmul(desc),
            OperationIr::Float(_, FloatOperationIr::Random(desc)) => self.random(&desc.out),
            OperationIr::Module(ModuleOperationIr::Linear(desc)) => self.linear(desc),
            OperationIr::Activation(
                ActivationOperationIr::Relu(desc) | ActivationOperationIr::Gelu(desc),
            ) => self.unary(Linearity::Nonlinear, desc),
            OperationIr::BaseFloat(op) | OperationIr::BaseInt(op) | OperationIr::BaseBool(op) => {
                match op {
                    BaseOperationIr::Empty(desc)
                    | BaseOperationIr::Ones(desc)
                    | BaseOperationIr::Zeros(desc) => self.created(&desc.out),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn numeric(&self, dtype: DType, op: &NumericOperationIr) -> Option<OpPlan> {
        match op {
            NumericOperationIr::Add(desc) | NumericOperationIr::Sub(desc) => {
                self.binary(Linearity::Linear, desc)
            }
            NumericOperationIr::Mul(desc) => self.binary(Linearity::Multilinear, desc),
            NumericOperationIr::Div(desc) => self.binary(Linearity::div(dtype), desc),
            NumericOperationIr::MulScalar(desc) => self.scalar(Linearity::Linear, desc),
            NumericOperationIr::DivScalar(desc) => self.scalar(Linearity::div(dtype), desc),
            NumericOperationIr::Neg(desc) => self.unary(Linearity::Linear, desc),
            NumericOperationIr::Sum(desc) => self.sum(&desc.input, &desc.out, Reduction::All),
            NumericOperationIr::SumDim(desc) => {
                self.sum(&desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            NumericOperationIr::Full(desc) => self.created(&desc.out),
            NumericOperationIr::IntRandom(desc) => self.random(&desc.out),
            _ => None,
        }
    }

    fn unary(&self, linearity: Linearity, desc: &UnaryOpIr) -> Option<OpPlan> {
        self.elementwise(linearity, [&desc.input], &desc.out)
    }

    fn scalar(&self, linearity: Linearity, desc: &ScalarOpIr) -> Option<OpPlan> {
        self.elementwise(linearity, [&desc.lhs], &desc.out)
    }

    fn binary(&self, linearity: Linearity, desc: &BinaryOpIr) -> Option<OpPlan> {
        self.elementwise(linearity, [&desc.lhs, &desc.rhs], &desc.out)
    }

    /// Broadcasting lines up dims of the same index, so an input with another number of dims has
    /// no rule.
    fn elementwise<const N: usize>(
        &self,
        linearity: Linearity,
        inputs: [&TensorIr; N],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let num_dims = out.shape.num_dims();
        if inputs
            .iter()
            .any(|input| input.shape.num_dims() != num_dims)
        {
            return None;
        }
        let placement = linearity.placement(
            inputs.map(|input| self.placement(input)),
            inputs.map(|input| &input.shape),
        );
        Targets::of(inputs, placement.inputs)
            .output(out, placement.output)
            .every_member()
    }

    fn matmul(&self, desc: &MatmulOpIr) -> Option<OpPlan> {
        let placement = MatmulRule::new(
            self.placement(&desc.lhs),
            self.placement(&desc.rhs),
            &desc.lhs.shape,
            &desc.rhs.shape,
        )
        .placement();
        self.ruled([&desc.lhs, &desc.rhs], &desc.out, placement)
    }

    fn linear(&self, desc: &LinearOpIr) -> Option<OpPlan> {
        let rule = LinearRule::new(
            self.placement(&desc.x),
            self.placement(&desc.weight),
            &desc.x.shape,
            &desc.weight.shape,
        );
        let OpPlacement {
            inputs: [x, weight, bias],
            output,
        } = rule.placement();
        let mut targets =
            Targets::of([&desc.x, &desc.weight], [x, weight]).output(&desc.out, output);
        let Some(bias_tensor) = &desc.bias else {
            return targets.every_member();
        };
        targets = targets.input(bias_tensor, bias);
        match rule.bias_on_one_member() {
            true => targets.build(Execution::BiasOnFirstMember),
            false => targets.every_member(),
        }
    }

    fn sum(&self, input: &TensorIr, out: &TensorIr, reduction: Reduction) -> Option<OpPlan> {
        let placement = ReduceRule::new(self.placement(input), reduction).placement();
        self.ruled([input], out, placement)
    }

    fn created(&self, out: &TensorIr) -> Option<OpPlan> {
        Targets::default()
            .output(out, GroupPlacement::Replicated)
            .every_member()
    }

    fn random(&self, out: &TensorIr) -> Option<OpPlan> {
        Targets::default()
            .output(out, GroupPlacement::Replicated)
            .build(Execution::FirstMemberCopied)
    }

    fn ruled<const N: usize>(
        &self,
        inputs: [&TensorIr; N],
        out: &TensorIr,
        placement: OpPlacement<N>,
    ) -> Option<OpPlan> {
        Targets::of(inputs, placement.inputs)
            .output(out, placement.output)
            .every_member()
    }

    fn placement(&self, tensor: &TensorIr) -> GroupPlacement {
        self.placements.get(&tensor.id)
    }
}

/// The placements an op's tensors take, refusing an input asked to be in two places at once.
#[derive(Default)]
struct Targets {
    inputs: HashMap<TensorId, GroupPlacement>,
    outputs: HashMap<TensorId, GroupPlacement>,
    conflict: bool,
}

impl Targets {
    fn of<const N: usize>(tensors: [&TensorIr; N], placements: [GroupPlacement; N]) -> Self {
        tensors
            .into_iter()
            .zip(placements)
            .fold(Self::default(), |targets, (tensor, placement)| {
                targets.input(tensor, placement)
            })
    }

    fn input(mut self, tensor: &TensorIr, placement: GroupPlacement) -> Self {
        let previous = self.inputs.insert(tensor.id, placement);
        self.conflict |= previous.is_some_and(|previous| previous != placement);
        self
    }

    fn output(mut self, tensor: &TensorIr, placement: GroupPlacement) -> Self {
        self.outputs.insert(tensor.id, placement);
        self
    }

    fn every_member(self) -> Option<OpPlan> {
        self.build(Execution::EveryMember)
    }

    fn build(self, execution: Execution) -> Option<OpPlan> {
        (!self.conflict).then_some(OpPlan {
            inputs: self.inputs,
            outputs: self.outputs,
            execution,
        })
    }
}
