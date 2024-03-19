use super::{BinaryOperator, ClampOperator, Item, Operation, Operator, UnaryOperator, Variable};

/// Define a vectorization scheme.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Default, Hash)]
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
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Operation::Operator(op) => Operation::Operator(op.vectorize(vectorization)),
            Operation::Procedure(op) => Operation::Procedure(op.vectorize(vectorization)),
            Operation::Metadata(_) => panic!(
                "Metadata can't be vectorized, they should only be generated after vectorization."
            ),
            Operation::Branch(_) => panic!(
                "A branch can't be vectorized, they should only be generated after vectorization."
            ),
            Operation::Synchronization(_) => panic!(
                "Synchronization instructions can't be vectorized, they should only be generated after vectorization."
            ),
        }
    }
}

impl Operator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Operator::Max(op) => Operator::Max(op.vectorize(vectorization)),
            Operator::Min(op) => Operator::Min(op.vectorize(vectorization)),
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
            Operator::Ceil(op) => Operator::Ceil(op.vectorize(vectorization)),
            Operator::Erf(op) => Operator::Erf(op.vectorize(vectorization)),
            Operator::Recip(op) => Operator::Recip(op.vectorize(vectorization)),
            Operator::Equal(op) => Operator::Equal(op.vectorize(vectorization)),
            Operator::NotEqual(op) => Operator::NotEqual(op.vectorize(vectorization)),
            Operator::Lower(op) => Operator::Lower(op.vectorize(vectorization)),
            Operator::Clamp(op) => Operator::Clamp(op.vectorize(vectorization)),
            Operator::Greater(op) => Operator::Greater(op.vectorize(vectorization)),
            Operator::LowerEqual(op) => Operator::LowerEqual(op.vectorize(vectorization)),
            Operator::GreaterEqual(op) => Operator::GreaterEqual(op.vectorize(vectorization)),
            Operator::Assign(op) => {
                if let Variable::GlobalScalar(_, _) = op.input {
                    // Assign will not change the type of the output if the input can't be
                    // vectorized.
                    return Operator::Assign(op.clone());
                }

                Operator::Assign(op.vectorize(vectorization))
            }
            Operator::Modulo(op) => Operator::Modulo(op.vectorize(vectorization)),
            Operator::IndexAssign(op) => Operator::IndexAssign(op.vectorize(vectorization)),
            Operator::And(op) => Operator::And(op.vectorize(vectorization)),
            Operator::Or(op) => Operator::Or(op.vectorize(vectorization)),
            Operator::Not(op) => Operator::Not(op.vectorize(vectorization)),
            Operator::BitwiseAnd(op) => Operator::BitwiseAnd(op.vectorize(vectorization)),
            Operator::BitwiseXor(op) => Operator::BitwiseXor(op.vectorize(vectorization)),
            Operator::ShiftLeft(op) => Operator::ShiftLeft(op.vectorize(vectorization)),
            Operator::ShiftRight(op) => Operator::ShiftRight(op.vectorize(vectorization)),
        }
    }
}

impl BinaryOperator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        let lhs = self.lhs.vectorize(vectorization);
        let rhs = self.rhs.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self { lhs, rhs, out }
    }
}

impl UnaryOperator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        let input = self.input.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self { input, out }
    }
}

impl ClampOperator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            input: self.input.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
            min_value: self.min_value.vectorize(vectorization),
            max_value: self.max_value.vectorize(vectorization),
        }
    }
}

impl Variable {
    pub(crate) fn vectorize(&self, vectorize: Vectorization) -> Self {
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
            Variable::SharedMemory(index, item, size) => Variable::SharedMemory(
                *index,
                item.vectorize(vectorize),
                item.vectorized_size(vectorize, *size),
            ),
            Variable::LocalArray(index, item, name, size) => Variable::LocalArray(
                *index,
                item.vectorize(vectorize),
                *name,
                item.vectorized_size(vectorize, *size),
            ),
            Variable::ConstantScalar(_, _) => *self,
            Variable::GlobalScalar(_, _) => *self,
            Variable::Id => *self,
            Variable::Rank => *self,
            Variable::LocalScalar(_, _, _) => *self,
            Variable::LocalInvocationIndex => *self,
            Variable::LocalInvocationIdX => *self,
            Variable::LocalInvocationIdY => *self,
            Variable::LocalInvocationIdZ => *self,
            Variable::WorkgroupIdX => *self,
            Variable::WorkgroupIdY => *self,
            Variable::WorkgroupIdZ => *self,
            Variable::GlobalInvocationIdX => *self,
            Variable::GlobalInvocationIdY => *self,
            Variable::GlobalInvocationIdZ => *self,
            Variable::WorkgroupSizeX => *self,
            Variable::WorkgroupSizeY => *self,
            Variable::WorkgroupSizeZ => *self,
            Variable::NumWorkgroupsX => *self,
            Variable::NumWorkgroupsY => *self,
            Variable::NumWorkgroupsZ => *self,
        }
    }
}

impl Item {
    pub(crate) fn vectorize(&self, vectorize: Vectorization) -> Item {
        match vectorize {
            Vectorization::Vec4 => Item::Vec4(self.elem()),
            Vectorization::Vec3 => Item::Vec3(self.elem()),
            Vectorization::Vec2 => Item::Vec2(self.elem()),
            Vectorization::Scalar => Item::Scalar(self.elem()),
        }
    }

    pub(crate) fn vectorized_size(&self, vectorize: Vectorization, size: u32) -> u32 {
        match vectorize {
            Vectorization::Vec4 => size / 4,
            Vectorization::Vec3 => size / 3,
            Vectorization::Vec2 => size / 2,
            Vectorization::Scalar => size,
        }
    }
}
