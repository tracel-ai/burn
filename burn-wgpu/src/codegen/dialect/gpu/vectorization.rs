use super::{
    BinaryOperation, ClampOperation, ConditionalAssignOperation, Item, Operation,
    ReadGlobalOperation, ReadGlobalWithLayoutOperation, UnaryOperation, Variable,
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
            Operation::Add(op) => Operation::Add(op.vectorize(vectorization)),
            Operation::Sub(op) => Operation::Sub(op.vectorize(vectorization)),
            Operation::Mul(op) => Operation::Mul(op.vectorize(vectorization)),
            Operation::Div(op) => Operation::Div(op.vectorize(vectorization)),
            Operation::Abs(op) => Operation::Abs(op.vectorize(vectorization)),
            Operation::Exp(op) => Operation::Exp(op.vectorize(vectorization)),
            Operation::Log(op) => Operation::Log(op.vectorize(vectorization)),
            Operation::Log1p(op) => Operation::Log1p(op.vectorize(vectorization)),
            Operation::Cos(op) => Operation::Cos(op.vectorize(vectorization)),
            Operation::Sin(op) => Operation::Sin(op.vectorize(vectorization)),
            Operation::Tanh(op) => Operation::Tanh(op.vectorize(vectorization)),
            Operation::Powf(op) => Operation::Powf(op.vectorize(vectorization)),
            Operation::Sqrt(op) => Operation::Sqrt(op.vectorize(vectorization)),
            Operation::Erf(op) => Operation::Erf(op.vectorize(vectorization)),
            Operation::Recip(op) => Operation::Recip(op.vectorize(vectorization)),
            Operation::Equal(op) => Operation::Equal(op.vectorize(vectorization)),
            Operation::Lower(op) => Operation::Lower(op.vectorize(vectorization)),
            Operation::Clamp(op) => Operation::Clamp(op.vectorize(vectorization)),
            Operation::Greater(op) => Operation::Greater(op.vectorize(vectorization)),
            Operation::LowerEqual(op) => Operation::LowerEqual(op.vectorize(vectorization)),
            Operation::GreaterEqual(op) => Operation::GreaterEqual(op.vectorize(vectorization)),
            Operation::ConditionalAssign(op) => {
                Operation::ConditionalAssign(op.vectorize(vectorization))
            }
            Operation::AssignGlobal(op) => Operation::AssignGlobal(op.vectorize(vectorization)),
            Operation::AssignLocal(op) => Operation::AssignLocal(op.vectorize(vectorization)),
            Operation::ReadGlobal(op) => Operation::ReadGlobal(op.vectorize(vectorization)),
            Operation::ReadGlobalWithLayout(op) => {
                Operation::ReadGlobalWithLayout(op.vectorize(vectorization))
            }
        }
    }
}

impl BinaryOperation {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let lhs = self.lhs.vectorize(vectorization);
        let rhs = self.rhs.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self { lhs, rhs, out }
    }
}

impl UnaryOperation {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let input = self.input.vectorize(vectorization);
        let out = self.out.vectorize(vectorization);

        Self { input, out }
    }
}

impl ClampOperation {
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

impl ConditionalAssignOperation {
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

impl ReadGlobalOperation {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        let variable = self.variable.vectorize(vectorization);

        Self { variable }
    }
}

impl ReadGlobalWithLayoutOperation {
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
            Variable::Local(index, item) => Variable::Local(*index, item.vectorize(vectorize)),
            Variable::Output(index, item) => Variable::Output(*index, item.vectorize(vectorize)),
            Variable::Constant(index, item) => {
                Variable::Constant(*index, item.vectorize(vectorize))
            }
            Variable::Scalar(index, item) => Variable::Scalar(*index, *item), // Don't vectorize
                                                                              // scalar variables.
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
