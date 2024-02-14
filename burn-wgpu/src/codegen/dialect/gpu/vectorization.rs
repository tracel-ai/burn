use super::{
    BinaryOperator, ClampOperator, ConditionalAssignOperator, Item, Operation, Operator,
    ReadGlobalOperator, ReadGlobalWithLayoutOperator, UnaryOperator, Variable,
};

/// Define a vectorization scheme.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
pub enum Vectorization {
    /// Use vec4 for vectorization.
    Vec4,
    /// Use vec3 for vectorization.
    Vec3,
    /// Use vec2 for vectorization.
    Vec2,
    /// Don't vectorize.
    Scalar,
}

impl Operation {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Operation::Operator(op) => Operation::Operator(op.vectorize(vectorization)),
            Operation::Algorithm(op) => Operation::Algorithm(op.clone()),
            Operation::Metadata(op) => Operation::Metadata(op.clone()),
            Operation::Loop(op) => Operation::Loop(op.clone()),
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
            Operator::AssignGlobal(op) => Operator::AssignGlobal(op.vectorize(vectorization)),
            Operator::AssignLocal(op) => Operator::AssignLocal(op.vectorize(vectorization)),
            Operator::ReadGlobal(op) => Operator::ReadGlobal(op.vectorize(vectorization)),
            Operator::ReadGlobalWithLayout(op) => {
                Operator::ReadGlobalWithLayout(op.vectorize(vectorization))
            }
            Operator::Modulo(op) => Operator::Modulo(op.vectorize(vectorization)),
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

impl ReadGlobalOperator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let variable = self.variable.vectorize(vectorization);

        Self { variable }
    }
}

impl ReadGlobalWithLayoutOperator {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let variable = self.variable.vectorize(vectorization);
        let tensor_read_pos = self.tensor_read_pos;
        let tensor_layout_pos = self.tensor_layout_pos;

        Self {
            variable,
            tensor_read_pos,
            tensor_layout_pos,
        }
    }
}

impl Variable {
    pub fn vectorize(&self, vectorize: Vectorization) -> Self {
        match self {
            Variable::Input(index, item) => Variable::Input(*index, item.vectorize(vectorize)),
            Variable::Local(index, item, name) => {
                Variable::Local(*index, item.vectorize(vectorize), name.clone())
            }
            Variable::Output(index, item) => Variable::Output(*index, item.vectorize(vectorize)),
            Variable::Constant(index, item) => {
                Variable::Constant(*index, item.vectorize(vectorize))
            }
            Variable::Scalar(index, item) => Variable::Scalar(*index, *item),
            Variable::Id => Variable::Id,
            Variable::Rank => Variable::Rank,
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
