use std::collections::HashMap;

use burn_ir::{
    ActivationOperationIr, BinaryOpIr, FloatOperationIr, LinearOpIr, MatmulOpIr, ModuleOperationIr,
    NumericOperationIr, OperationIr, ScalarOpIr, TensorId, TensorIr, UnaryOpIr,
};

use super::sharding_map::ShardingMap;
use crate::{LinearLayerRule, Linearity, MatmulRule, OpSharding, Sharding, SumRule, SummedDims};

/// Where each tensor of one op must be for the op to run on every member, where its outputs land,
/// and what each member runs.
#[derive(Debug)]
pub struct OpPlan {
    input_shardings: HashMap<TensorId, Sharding>,
    output_shardings: HashMap<TensorId, Sharding>,
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
    /// An op without a rule, or whose inputs would need two shardings at once, gathers every
    /// input and runs whole on every member.
    pub fn new(op: &OperationIr, shardings: &ShardingMap) -> Self {
        Analysis { shardings }
            .plan(op)
            .unwrap_or_else(|| Self::gathered(op))
    }

    pub fn input_sharding(&self, id: &TensorId) -> Sharding {
        self.input_shardings[id]
    }

    pub fn output_shardings(&self) -> &HashMap<TensorId, Sharding> {
        &self.output_shardings
    }

    pub fn execution(&self) -> Execution {
        self.execution
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

struct Analysis<'a> {
    shardings: &'a ShardingMap,
}

impl Analysis<'_> {
    fn plan(&self, op: &OperationIr) -> Option<OpPlan> {
        match op {
            OperationIr::NumericFloat(_, op) => self.numeric(op),
            OperationIr::Float(_, FloatOperationIr::Matmul(desc)) => self.matmul(desc),
            OperationIr::Float(_, FloatOperationIr::Random(desc)) => self.random(&desc.out),
            OperationIr::NumericInt(_, NumericOperationIr::IntRandom(desc)) => {
                self.random(&desc.out)
            }
            OperationIr::Module(ModuleOperationIr::Linear(desc)) => self.linear(desc),
            OperationIr::Activation(
                ActivationOperationIr::Relu(desc) | ActivationOperationIr::Gelu(desc),
            ) => self.unary(Linearity::Nonlinear, desc),
            _ => None,
        }
    }

    fn numeric(&self, op: &NumericOperationIr) -> Option<OpPlan> {
        match op {
            NumericOperationIr::Add(desc) | NumericOperationIr::Sub(desc) => {
                self.binary(Linearity::Linear, desc)
            }
            NumericOperationIr::Mul(desc) => self.binary(Linearity::Multilinear, desc),
            NumericOperationIr::Div(desc) => self.binary(Linearity::LinearIn { input: 0 }, desc),
            NumericOperationIr::MulScalar(desc) | NumericOperationIr::DivScalar(desc) => {
                self.scalar(Linearity::Linear, desc)
            }
            NumericOperationIr::Neg(desc) => self.unary(Linearity::Linear, desc),
            NumericOperationIr::Sum(desc) => self.sum(&desc.input, &desc.out, SummedDims::All),
            NumericOperationIr::SumDim(desc) => {
                self.sum(&desc.input, &desc.out, SummedDims::Dim(desc.axis))
            }
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
        let sharding = linearity.sharding(
            inputs.map(|input| self.sharding(input)),
            inputs.map(|input| &input.shape),
        );
        TargetShardings::of(inputs, sharding.inputs)
            .output(out, sharding.output)
            .every_member()
    }

    fn matmul(&self, desc: &MatmulOpIr) -> Option<OpPlan> {
        let sharding = MatmulRule::new(
            self.sharding(&desc.lhs),
            self.sharding(&desc.rhs),
            desc.lhs.shape.num_dims(),
        )
        .sharding();
        self.ruled([&desc.lhs, &desc.rhs], &desc.out, sharding)
    }

    fn linear(&self, desc: &LinearOpIr) -> Option<OpPlan> {
        let rule = LinearLayerRule::new(
            self.sharding(&desc.x),
            self.sharding(&desc.weight),
            desc.x.shape.num_dims(),
        );
        let OpSharding {
            inputs: [x, weight, bias],
            output,
        } = rule.sharding();
        let mut targets =
            TargetShardings::of([&desc.x, &desc.weight], [x, weight]).output(&desc.out, output);
        let Some(bias_tensor) = &desc.bias else {
            return targets.every_member();
        };
        targets = targets.input(bias_tensor, bias);
        match rule.bias_on_one_member() {
            true => targets.build(Execution::BiasOnFirstMember),
            false => targets.every_member(),
        }
    }

    fn sum(&self, input: &TensorIr, out: &TensorIr, summed_dims: SummedDims) -> Option<OpPlan> {
        let sharding = SumRule::new(self.sharding(input), summed_dims).sharding();
        self.ruled([input], out, sharding)
    }

    fn random(&self, out: &TensorIr) -> Option<OpPlan> {
        TargetShardings::default()
            .output(out, Sharding::Replicated)
            .build(Execution::FirstMemberCopied)
    }

    fn ruled<const N: usize>(
        &self,
        inputs: [&TensorIr; N],
        out: &TensorIr,
        sharding: OpSharding<N>,
    ) -> Option<OpPlan> {
        TargetShardings::of(inputs, sharding.inputs)
            .output(out, sharding.output)
            .every_member()
    }

    fn sharding(&self, tensor: &TensorIr) -> Sharding {
        self.shardings.get(&tensor.id)
    }
}

/// The shardings an op's tensors take, refusing an input asked for two at once.
#[derive(Default)]
struct TargetShardings {
    inputs: HashMap<TensorId, Sharding>,
    outputs: HashMap<TensorId, Sharding>,
    conflict: bool,
}

impl TargetShardings {
    fn of<const N: usize>(tensors: [&TensorIr; N], shardings: [Sharding; N]) -> Self {
        tensors
            .into_iter()
            .zip(shardings)
            .fold(Self::default(), |targets, (tensor, sharding)| {
                targets.input(tensor, sharding)
            })
    }

    fn input(mut self, tensor: &TensorIr, sharding: Sharding) -> Self {
        let previous = self.inputs.insert(tensor.id, sharding);
        self.conflict |= previous.is_some_and(|previous| previous != sharding);
        self
    }

    fn output(mut self, tensor: &TensorIr, sharding: Sharding) -> Self {
        self.outputs.insert(tensor.id, sharding);
        self
    }

    fn every_member(self) -> Option<OpPlan> {
        self.build(Execution::EveryMember)
    }

    fn build(self, execution: Execution) -> Option<OpPlan> {
        (!self.conflict).then_some(OpPlan {
            input_shardings: self.inputs,
            output_shardings: self.outputs,
            execution,
        })
    }
}
