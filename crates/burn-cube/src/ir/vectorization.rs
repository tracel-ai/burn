use super::{
    BinaryOperator, ClampOperator, FmaOperator, InitOperator, Item, Operation, Operator,
    SliceOperator, Subcube, UnaryOperator, Variable,
};

pub type Vectorization = u8;

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
            Operation::Subcube(op) => Operation::Subcube(op.vectorize(vectorization)),
            Operation::CoopMma(_) => panic!(
                "Cooperative matrix-multiply and accumulate doesn't support vectorization."
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
            Operator::Fma(op) => Operator::Fma(op.vectorize(vectorization)),
            Operator::Index(op) => Operator::Index(op.vectorize(vectorization)),
            Operator::UncheckedIndex(op) => Operator::UncheckedIndex(op.vectorize(vectorization)),
            Operator::Sub(op) => Operator::Sub(op.vectorize(vectorization)),
            Operator::Mul(op) => Operator::Mul(op.vectorize(vectorization)),
            Operator::Div(op) => Operator::Div(op.vectorize(vectorization)),
            Operator::Floor(op) => Operator::Floor(op.vectorize(vectorization)),
            Operator::Ceil(op) => Operator::Ceil(op.vectorize(vectorization)),
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
            Operator::NotEqual(op) => Operator::NotEqual(op.vectorize(vectorization)),
            Operator::Lower(op) => Operator::Lower(op.vectorize(vectorization)),
            Operator::Clamp(op) => Operator::Clamp(op.vectorize(vectorization)),
            Operator::Greater(op) => Operator::Greater(op.vectorize(vectorization)),
            Operator::LowerEqual(op) => Operator::LowerEqual(op.vectorize(vectorization)),
            Operator::GreaterEqual(op) => Operator::GreaterEqual(op.vectorize(vectorization)),
            Operator::Assign(op) => {
                if let Variable::GlobalScalar { .. } = op.input {
                    // Assign will not change the type of the output if the input can't be
                    // vectorized.
                    return Operator::Assign(op.clone());
                }

                Operator::Assign(op.vectorize(vectorization))
            }
            Operator::Modulo(op) => Operator::Modulo(op.vectorize(vectorization)),
            Operator::IndexAssign(op) => Operator::IndexAssign(op.vectorize(vectorization)),
            Operator::UncheckedIndexAssign(op) => {
                Operator::UncheckedIndexAssign(op.vectorize(vectorization))
            }
            Operator::And(op) => Operator::And(op.vectorize(vectorization)),
            Operator::Or(op) => Operator::Or(op.vectorize(vectorization)),
            Operator::Not(op) => Operator::Not(op.vectorize(vectorization)),
            Operator::BitwiseAnd(op) => Operator::BitwiseAnd(op.vectorize(vectorization)),
            Operator::BitwiseXor(op) => Operator::BitwiseXor(op.vectorize(vectorization)),
            Operator::ShiftLeft(op) => Operator::ShiftLeft(op.vectorize(vectorization)),
            Operator::ShiftRight(op) => Operator::ShiftRight(op.vectorize(vectorization)),
            Operator::Remainder(op) => Operator::Remainder(op.vectorize(vectorization)),
            Operator::Slice(op) => Operator::Slice(op.vectorize(vectorization)),
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

impl SliceOperator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        let input = self.input.vectorize(vectorization);
        let start = self.start.vectorize(vectorization);
        let end = self.end.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self {
            input,
            start,
            end,
            out,
        }
    }
}

impl InitOperator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        let out = self.out.vectorize(vectorization);

        Self { out }
    }
}

impl Subcube {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Subcube::Elect(op) => Subcube::Elect(op.vectorize(vectorization)),
            Subcube::All(op) => Subcube::All(op.vectorize(vectorization)),
            Subcube::Any(op) => Subcube::Any(op.vectorize(vectorization)),
            Subcube::Broadcast(op) => Subcube::Broadcast(op.vectorize(vectorization)),
            Subcube::Sum(op) => Subcube::Sum(op.vectorize(vectorization)),
            Subcube::Prod(op) => Subcube::Prod(op.vectorize(vectorization)),
            Subcube::And(op) => Subcube::And(op.vectorize(vectorization)),
            Subcube::Or(op) => Subcube::Or(op.vectorize(vectorization)),
            Subcube::Xor(op) => Subcube::Xor(op.vectorize(vectorization)),
            Subcube::Min(op) => Subcube::Min(op.vectorize(vectorization)),
            Subcube::Max(op) => Subcube::Max(op.vectorize(vectorization)),
        }
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

impl FmaOperator {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Self {
        Self {
            a: self.a.vectorize(vectorization),
            b: self.b.vectorize(vectorization),
            c: self.c.vectorize(vectorization),
            out: self.out.vectorize(vectorization),
        }
    }
}

impl Variable {
    pub(crate) fn vectorize(&self, vectorize: Vectorization) -> Self {
        match self {
            Variable::GlobalInputArray { id, item } => Variable::GlobalInputArray {
                id: *id,
                item: item.vectorize(vectorize),
            },
            Variable::Local { id, item, depth } => Variable::Local {
                id: *id,
                item: item.vectorize(vectorize),
                depth: *depth,
            },
            Variable::Slice { id, item, depth } => Variable::Slice {
                id: *id,
                item: item.vectorize(vectorize),
                depth: *depth,
            },
            Variable::GlobalOutputArray { id, item } => Variable::GlobalOutputArray {
                id: *id,
                item: item.vectorize(vectorize),
            },
            Variable::SharedMemory { id, item, length } => Variable::SharedMemory {
                id: *id,
                item: item.vectorize(vectorize),
                length: item.vectorized_size(vectorize, *length),
            },
            Variable::LocalArray {
                id,
                item,
                depth,
                length,
            } => Variable::LocalArray {
                id: *id,
                item: item.vectorize(vectorize),
                depth: *depth,
                length: item.vectorized_size(vectorize, *length),
            },
            Variable::ConstantScalar { .. } => *self,
            Variable::GlobalScalar { .. } => *self,
            Variable::AbsolutePos => *self,
            Variable::Rank => *self,
            Variable::LocalScalar { .. } => *self,
            Variable::Matrix { .. } => *self,
            Variable::UnitPos => *self,
            Variable::UnitPosX => *self,
            Variable::UnitPosY => *self,
            Variable::UnitPosZ => *self,
            Variable::CubePosX => *self,
            Variable::CubePosY => *self,
            Variable::CubePosZ => *self,
            Variable::AbsolutePosX => *self,
            Variable::AbsolutePosY => *self,
            Variable::AbsolutePosZ => *self,
            Variable::CubeDimX => *self,
            Variable::CubeDimY => *self,
            Variable::CubeDimZ => *self,
            Variable::CubeCountX => *self,
            Variable::CubeCountY => *self,
            Variable::CubeCountZ => *self,
            Variable::CubePos => *self,
            Variable::CubeCount => *self,
            Variable::CubeDim => *self,
            Variable::SubcubeDim => *self,
        }
    }
}

impl Item {
    pub(crate) fn vectorize(&self, vectorization: Vectorization) -> Item {
        Item {
            elem: self.elem,
            vectorization,
        }
    }

    pub(crate) fn vectorized_size(&self, vectorize: Vectorization, size: u32) -> u32 {
        size / (vectorize as u32)
    }
}
