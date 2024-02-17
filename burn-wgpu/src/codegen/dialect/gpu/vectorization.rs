use super::{
    Algorithm, BinaryOperator, ClampOperator, ConditionalAssignOperator, Item, MatmulAlgo,
    Operation, Operator, ReadGlobalAlgo, ReadGlobalWithLayoutAlgo, UnaryOperator, Variable,
    WriteGlobalAlgo,
};

/// Define a vectorization scheme.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default)]
pub enum Vectorization {
    /// Use vec4 for vectorization.
    Vec4,
    /// Use vec3 for vectorization.
    Vec3,
    /// Use vec2 for vectorization.
    Vec2,
    /// Don't vectorize.
    #[default]
    Scalar,
}

impl Operation {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Operation::Operator(op) => Operation::Operator(op.vectorize(vectorization)),
            Operation::Algorithm(op) => Operation::Algorithm(op.vectorize(vectorization)),
            Operation::Metadata(_) => panic!(
                "Metadata can't be vectorized, they should only be generated after vectorization."
            ),
            Operation::Loop(_) => panic!(
                "Loops can't be vectorized, they should only be generated after vectorization."
            ),
            Operation::Branch(_) => panic!(
                "A branch can't be vectorized, they should only be generated after vectorization."
            ),
        }
    }
}

impl Algorithm {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Algorithm::ReadGlobalWithLayout(op) => {
                Algorithm::ReadGlobalWithLayout(op.vectorize(vectorization))
            }
            Algorithm::ReadGlobal(op) => Algorithm::ReadGlobal(op.vectorize(vectorization)),
            Algorithm::Matmul(op) => Algorithm::Matmul(op.vectorize(vectorization)),
            Algorithm::WriteGlobal(op) => Algorithm::WriteGlobal(op.vectorize(vectorization)),
        }
    }
}

impl ReadGlobalWithLayoutAlgo {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            global: self.global.vectorize(vectorization),
            layout: self.layout.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
        }
    }
}

impl ReadGlobalAlgo {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            global: self.global.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
        }
    }
}
impl WriteGlobalAlgo {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            input: self.input.vectorize(vectorization),
            global: self.global.vectorize(vectorization),
        }
    }
}

impl MatmulAlgo {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            MatmulAlgo::MemCoalescing {
                variables,
                block_size,
            } => MatmulAlgo::MemCoalescing {
                variables: variables.vectorize(vectorization),
                block_size: *block_size,
            },
        }
    }
}
impl Operator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Operator::Add(op) => Operator::Add(op.vectorize(vectorization)),
            Operator::Index(op) => Operator::Index(op.vectorize(vectorization)),
            Operator::Sub(op) => Operator::Sub(op.vectorize(vectorization)),
            Operator::Mul(op) => Operator::Mul(op.vectorize(vectorization)),
            Operator::Div(op) => Operator::Div(op.vectorize(vectorization)),
            Operator::Abs(op) => Operator::Abs(op.vectorize(vectorization)),
            Operator::Exp(op) => Operator::Exp(op.vectorize(vectorization)),
            Operator::Log(op) => Operator::Log(op.vectorize(vectorization)),
            Operator::Log1p(op) => Operator::Log1p(op.vectorize(vectorization)),
            Operator::Cos(op) => Operator::Cos(op.vectorize(vectorization)),
            Operator::Sin(op) => Operator::Sin(op.vectorize(vectorization)),
            Operator::Tanh(op) => Operator::Tanh(op.vectorize(vectorization)),
            Operator::Powf(op) => Operator::Powf(op.vectorize(vectorization)),
            Operator::Sqrt(op) => Operator::Sqrt(op.vectorize(vectorization)),
            Operator::Erf(op) => Operator::Erf(op.vectorize(vectorization)),
            Operator::Recip(op) => Operator::Recip(op.vectorize(vectorization)),
            Operator::Equal(op) => Operator::Equal(op.vectorize(vectorization)),
            Operator::Lower(op) => Operator::Lower(op.vectorize(vectorization)),
            Operator::Clamp(op) => Operator::Clamp(op.vectorize(vectorization)),
            Operator::Greater(op) => Operator::Greater(op.vectorize(vectorization)),
            Operator::LowerEqual(op) => Operator::LowerEqual(op.vectorize(vectorization)),
            Operator::GreaterEqual(op) => Operator::GreaterEqual(op.vectorize(vectorization)),
            Operator::ConditionalAssign(op) => {
                Operator::ConditionalAssign(op.vectorize(vectorization))
            }
            Operator::AssignLocal(op) => {
                if let Variable::GlobalScalar(_, _) = op.input {
                    // Assign will not change the type of the output if the input can't be
                    // vectorized.
                    return Operator::AssignLocal(op.clone());
                }

                Operator::AssignLocal(op.vectorize(vectorization))
            }
            Operator::Modulo(op) => Operator::Modulo(op.vectorize(vectorization)),
            Operator::IndexAssign(op) => Operator::IndexAssign(op.vectorize(vectorization)),
        }
    }
}

impl BinaryOperator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let lhs = self.lhs.vectorize(vectorization);
        let rhs = self.rhs.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self { lhs, rhs, out }
    }
}

impl UnaryOperator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let input = self.input.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self { input, out }
    }
}

impl ClampOperator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let input = self.input.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);
        let min_value = self.min_value.vectorize(vectorization);
        let max_value = self.max_value.vectorize(vectorization);

        Self {
            input,
            out,
            min_value,
            max_value,
        }
    }
}

impl ConditionalAssignOperator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let cond = self.cond.vectorize(vectorization);
        let lhs = self.lhs.vectorize(vectorization);
        let rhs = self.rhs.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self {
            cond,
            lhs,
            rhs,
            out,
        }
    }
}

impl Variable {
    pub fn vectorize(&self, vectorize: Vectorization) -> Self {
        match self {
            Variable::GlobalInputArray(index, item) => {
                Variable::GlobalInputArray(*index, item.vectorize(vectorize))
            }
            Variable::Local(index, item, name) => {
                Variable::Local(*index, item.vectorize(vectorize), *name)
            }
            Variable::GlobalOutputArray(index, item) => {
                Variable::GlobalOutputArray(*index, item.vectorize(vectorize))
            }
            Variable::ConstantScalar(_, _) => *self,
            Variable::GlobalScalar(_, _) => *self,
            Variable::Id => *self,
            Variable::Rank => *self,
            Variable::LocalScalar(_, _, _) => *self,
            Variable::InvocationIndex => *self,
            Variable::WorkgroupIdX => *self,
            Variable::WorkgroupIdY => *self,
            Variable::WorkgroupIdZ => *self,
            Variable::GlobalInvocationIdX => *self,
            Variable::GlobalInvocationIdY => *self,
            Variable::GlobalInvocationIdZ => *self,
        }
    }
}

impl Item {
    pub fn vectorize(&self, vectorize: Vectorization) -> Item {
        match vectorize {
            Vectorization::Vec4 => Item::Vec4(self.elem()),
            Vectorization::Vec3 => Item::Vec3(self.elem()),
            Vectorization::Vec2 => Item::Vec2(self.elem()),
            Vectorization::Scalar => Item::Scalar(self.elem()),
        }
    }
}
