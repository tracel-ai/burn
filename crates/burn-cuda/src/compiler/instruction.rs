use super::{binary::*, unary::*, Component, Variable};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct BinaryInstruction {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone)]
pub struct UnaryInstruction {
    pub input: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    ArrayLength {
        input: Variable,
        out: Variable,
        num_inputs: usize,
        num_outputs: usize,
    },
    DeclareVariable {
        var: Variable,
    },
    Modulo(BinaryInstruction),
    Add(BinaryInstruction),
    Div(BinaryInstruction),
    Mul(BinaryInstruction),
    Sub(BinaryInstruction),
    Index(BinaryInstruction),
    IndexAssign(BinaryInstruction),
    CheckedIndexAssign(BinaryInstruction),
    Assign(UnaryInstruction),
    RangeLoop {
        i: Variable,
        start: Variable,
        end: Variable,
        instructions: Vec<Self>,
    },
    Loop {
        instructions: Vec<Self>,
    },
    If {
        cond: Variable,
        instructions: Vec<Self>,
    },
    IfElse {
        cond: Variable,
        instructions_if: Vec<Self>,
        instructions_else: Vec<Self>,
    },
    Return,
    Break,
    Stride {
        dim: Variable,
        position: usize,
        out: Variable,
    },
    Shape {
        dim: Variable,
        position: usize,
        out: Variable,
    },
    Equal(BinaryInstruction),
    NotEqual(BinaryInstruction),
    Lower(BinaryInstruction),
    Greater(BinaryInstruction),
    LowerEqual(BinaryInstruction),
    GreaterEqual(BinaryInstruction),
    Erf(UnaryInstruction),
    BitwiseAnd(BinaryInstruction),
    BitwiseXor(BinaryInstruction),
    ShiftLeft(BinaryInstruction),
    ShiftRight(BinaryInstruction),
    Abs(UnaryInstruction),
    Exp(UnaryInstruction),
    Log(UnaryInstruction),
    Log1p(UnaryInstruction),
    Cos(UnaryInstruction),
    Sin(UnaryInstruction),
    Tanh(UnaryInstruction),
    Powf(BinaryInstruction),
    Sqrt(UnaryInstruction),
    Min(BinaryInstruction),
    Max(BinaryInstruction),
    Not(UnaryInstruction),
    Or(BinaryInstruction),
    And(BinaryInstruction),
    Clamp {
        input: Variable,
        min_value: Variable,
        max_value: Variable,
        out: Variable,
    },
    SyncThreads,
    Ceil(UnaryInstruction),
    Floor(UnaryInstruction),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Return => f.write_str("return;"),
            Instruction::Break => f.write_str("break;"),
            Instruction::DeclareVariable { var } => {
                let item = var.item();
                f.write_fmt(format_args!("{item} {var};\n"))
            }
            Instruction::Add(it) => Add::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Mul(it) => Mul::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Div(it) => Div::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Sub(it) => Sub::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Modulo(inst) => Modulo::format(f, &inst.lhs, &inst.rhs, &inst.out),
            Instruction::BitwiseAnd(it) => BitwiseAnd::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::BitwiseXor(it) => BitwiseXor::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::ShiftLeft(it) => ShiftLeft::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::ShiftRight(it) => ShiftRight::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Index(it) => Index::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::IndexAssign(it) => IndexAssign::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::CheckedIndexAssign(it) => {
                IndexAssign::format(f, &it.lhs, &it.rhs, &it.out)
            }
            Instruction::Assign(it) => Assign::format(f, &it.input, &it.out),
            Instruction::RangeLoop {
                i,
                start,
                end,
                instructions,
            } => {
                f.write_fmt(format_args!(
                    "
for (uint {i} = {start}; {i} < {end}; {i}++) {{
"
                ))?;
                for instruction in instructions {
                    f.write_fmt(format_args!("{instruction}"))?;
                }

                f.write_str("}\n")
            }

            Instruction::Loop { instructions } => {
                f.write_fmt(format_args!("while (true) {{\n"))?;
                for i in instructions {
                    f.write_fmt(format_args!("{i}"))?;
                }
                f.write_str("}\n")
            }
            Instruction::If { cond, instructions } => {
                f.write_fmt(format_args!("if ({cond}) {{\n"))?;
                for i in instructions {
                    f.write_fmt(format_args!("{i}"))?;
                }
                f.write_str("}\n")
            }
            Instruction::IfElse {
                cond,
                instructions_if,
                instructions_else,
            } => {
                f.write_fmt(format_args!("if ({cond}) {{\n"))?;
                for i in instructions_if {
                    f.write_fmt(format_args!("{i}"))?;
                }
                f.write_str("} else {\n")?;
                for i in instructions_else {
                    f.write_fmt(format_args!("{i}"))?;
                }
                f.write_str("}\n")
            }
            Instruction::Stride { dim, position, out } => f.write_fmt(format_args!(
                "{out} = info[({position} * rank_2) + {dim} + 1];\n"
            )),
            Instruction::Shape { dim, position, out } => f.write_fmt(format_args!(
                "{out} = info[({position} * rank_2) + rank + {dim} + 1];\n"
            )),
            Instruction::Equal(it) => Equal::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::NotEqual(it) => NotEqual::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Lower(it) => Lower::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Greater(it) => Greater::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::LowerEqual(it) => LowerEqual::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::GreaterEqual(it) => GreaterEqual::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Erf(it) => Erf::format(f, &it.input, &it.out),
            Instruction::Abs(it) => Abs::format(f, &it.input, &it.out),
            Instruction::Exp(it) => Exp::format(f, &it.input, &it.out),
            Instruction::Log(it) => Log::format(f, &it.input, &it.out),
            Instruction::Log1p(it) => Log1p::format(f, &it.input, &it.out),
            Instruction::Cos(it) => Cos::format(f, &it.input, &it.out),
            Instruction::Sin(it) => Sin::format(f, &it.input, &it.out),
            Instruction::Tanh(it) => Tanh::format(f, &it.input, &it.out),
            Instruction::Powf(it) => Powf::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Sqrt(it) => Sqrt::format(f, &it.input, &it.out),
            Instruction::Max(it) => Max::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Min(it) => Min::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Not(it) => Not::format(f, &it.input, &it.out),
            Instruction::Or(it) => Or::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::And(it) => And::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => f.write_fmt(format_args!(
                "
{out} = min({input}, {max_value});
{out} = max({out}, {min_value});
                "
            )),
            Instruction::SyncThreads => f.write_str("__syncthreads();\n"),
            Instruction::Ceil(it) => Ceil::format(f, &it.input, &it.out),
            Instruction::Floor(it) => Floor::format(f, &it.input, &it.out),
            Instruction::ArrayLength {
                input,
                out,
                num_inputs,
                num_outputs,
            } => {
                let offset = num_inputs + num_outputs;
                let index = match input {
                    Variable::GlobalInputArray(index, _) => *index as usize,
                    Variable::GlobalOutputArray(index, _) => *index as usize + num_inputs,
                    _ => panic!("Can only know the len of a global array."),
                } + 1;
                f.write_fmt(format_args!(
                    "{out} = info[({offset} * 2 * info[0]) + {index}];\n"
                ))
            }
        }
    }
}
