use std::collections::HashMap;

use burn_ir::{
    ActivationOperationIr, BaseOperationIr, BinaryOpIr, FloatOperationIr, MatmulOpIr,
    NumericOperationIr, OperationIr, ScalarOpIr, ShapeOpIr, TensorId, TensorIr, UnaryOpIr,
};

use crate::{ElementwiseRule, MatmulRule, OpSharding, ReducedDims, ReshapeRule, Sharding, SumRule};

/// Where each tensor of one op must be for the op to run on every member, where its outputs land,
/// and what each member runs.
#[derive(Debug)]
pub struct OpPlan {
    input_shardings: HashMap<TensorId, Sharding>,
    output_shardings: HashMap<TensorId, Sharding>,
    execution: Execution,
}

/// What each member runs for one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    /// Each member runs the op on its own shards.
    EveryMember,
    /// Member 0 runs the op and the others take a copy, so random values agree.
    FirstMemberCopied,
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two shardings at once, gathers every
    /// input and runs whole on every member.
    pub fn new(
        op: &OperationIr,
        shardings: &HashMap<TensorId, Sharding>,
        num_members: usize,
    ) -> Self {
        GroupShardings {
            shardings,
            num_members,
        }
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

    /// `None` when one tensor, read as two inputs, is asked for two shardings.
    fn ruled<const N: usize>(
        inputs: [&TensorIr; N],
        out: &TensorIr,
        sharding: OpSharding<N>,
        execution: Execution,
    ) -> Option<Self> {
        let mut input_shardings = HashMap::new();
        for (tensor, target) in inputs.into_iter().zip(sharding.inputs) {
            if input_shardings
                .insert(tensor.id, target)
                .is_some_and(|previous| previous != target)
            {
                return None;
            }
        }
        Some(Self {
            input_shardings,
            output_shardings: HashMap::from([(out.id, sharding.output)]),
            execution,
        })
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

/// The shardings of a group's tensors, which pick the rule each op runs by.
struct GroupShardings<'a> {
    shardings: &'a HashMap<TensorId, Sharding>,
    num_members: usize,
}

impl GroupShardings<'_> {
    fn plan(&self, op: &OperationIr) -> Option<OpPlan> {
        match op {
            OperationIr::BaseFloat(BaseOperationIr::Reshape(desc))
            | OperationIr::BaseInt(BaseOperationIr::Reshape(desc))
            | OperationIr::BaseBool(BaseOperationIr::Reshape(desc)) => self.reshape(desc),
            OperationIr::NumericFloat(_, op) => self.numeric(op),
            OperationIr::Float(_, FloatOperationIr::Matmul(desc)) => self.matmul(desc),
            OperationIr::Float(_, FloatOperationIr::Random(desc)) => self.random(&desc.out),
            OperationIr::NumericInt(_, NumericOperationIr::IntRandom(desc)) => {
                self.random(&desc.out)
            }
            OperationIr::Activation(
                ActivationOperationIr::Relu(desc)
                | ActivationOperationIr::Gelu(desc)
                | ActivationOperationIr::Sigmoid(desc),
            ) => self.unary(ElementwiseRule::Nonlinear, desc),
            _ => None,
        }
    }

    fn numeric(&self, op: &NumericOperationIr) -> Option<OpPlan> {
        match op {
            NumericOperationIr::Add(desc) | NumericOperationIr::Sub(desc) => {
                self.binary(ElementwiseRule::Linear, desc)
            }
            NumericOperationIr::Mul(desc) => self.binary(ElementwiseRule::Multilinear, desc),
            NumericOperationIr::Div(desc) => {
                self.binary(ElementwiseRule::LinearIn { input: 0 }, desc)
            }
            NumericOperationIr::MulScalar(desc) | NumericOperationIr::DivScalar(desc) => {
                self.scalar(ElementwiseRule::Linear, desc)
            }
            NumericOperationIr::Neg(desc) => self.unary(ElementwiseRule::Linear, desc),
            NumericOperationIr::Sum(desc) => self.sum(&desc.input, &desc.out, ReducedDims::All),
            NumericOperationIr::SumDim(desc) => {
                self.sum(&desc.input, &desc.out, ReducedDims::Dim(desc.axis))
            }
            _ => None,
        }
    }

    fn unary(&self, rule: ElementwiseRule, desc: &UnaryOpIr) -> Option<OpPlan> {
        self.elementwise(rule, [&desc.input], &desc.out)
    }

    fn scalar(&self, rule: ElementwiseRule, desc: &ScalarOpIr) -> Option<OpPlan> {
        self.elementwise(rule, [&desc.lhs], &desc.out)
    }

    fn binary(&self, rule: ElementwiseRule, desc: &BinaryOpIr) -> Option<OpPlan> {
        self.elementwise(rule, [&desc.lhs, &desc.rhs], &desc.out)
    }

    /// Broadcasting lines up dims of the same index, so an input with another number of dims has
    /// no rule.
    fn elementwise<const N: usize>(
        &self,
        rule: ElementwiseRule,
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
        let sharding = rule.sharding(
            inputs.map(|input| self.sharding(input)),
            inputs.map(|input| &input.shape),
        );
        OpPlan::ruled(inputs, out, sharding, Execution::EveryMember)
    }

    fn matmul(&self, desc: &MatmulOpIr) -> Option<OpPlan> {
        let sharding = MatmulRule::new(
            self.sharding(&desc.lhs),
            self.sharding(&desc.rhs),
            desc.lhs.shape.num_dims(),
        )
        .sharding();
        OpPlan::ruled(
            [&desc.lhs, &desc.rhs],
            &desc.out,
            sharding,
            Execution::EveryMember,
        )
    }

    fn sum(&self, input: &TensorIr, out: &TensorIr, reduced_dims: ReducedDims) -> Option<OpPlan> {
        let sharding = SumRule::new(self.sharding(input), reduced_dims).sharding();
        OpPlan::ruled([input], out, sharding, Execution::EveryMember)
    }

    fn reshape(&self, desc: &ShapeOpIr) -> Option<OpPlan> {
        let sharding = ReshapeRule::new(
            self.sharding(&desc.input),
            &desc.input.shape,
            &desc.out.shape,
            self.num_members,
        )
        .sharding();
        OpPlan::ruled([&desc.input], &desc.out, sharding, Execution::EveryMember)
    }

    fn random(&self, out: &TensorIr) -> Option<OpPlan> {
        let sharding = OpSharding {
            inputs: [],
            output: Sharding::Replicated,
        };
        OpPlan::ruled([], out, sharding, Execution::FirstMemberCopied)
    }

    fn sharding(&self, tensor: &TensorIr) -> Sharding {
        self.shardings[&tensor.id]
    }
}
