use super::{binary::*, unary::*, Component, Variable, WarpInstruction, WmmaInstruction};
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
    Length {
        input: Variable,
        out: Variable,
        num_inputs: usize,
        num_outputs: usize,
    },
    SliceLength {
        input: Variable,
        out: Variable,
    },
    DeclareVariable {
        var: Variable,
    },
    Modulo(BinaryInstruction),
    Add(BinaryInstruction),
    Fma {
        a: Variable,
        b: Variable,
        c: Variable,
        out: Variable,
    },
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
    Slice {
        input: Variable,
        start: Variable,
        end: Variable,
        out: Variable,
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
    Wrap(WarpInstruction),
    Wmma(WmmaInstruction),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Return => f.write_str("return;"),
            Instruction::Break => f.write_str("break;"),
            Instruction::DeclareVariable { var } => match var {
                Variable::WmmaFragment { id: _, frag } => {
                    f.write_fmt(format_args!("{frag} {var};\n"))
                }
                _ => {
                    let item = var.item();
                    f.write_fmt(format_args!("{item} {var};\n"))
                }
            },
            Instruction::Add(it) => Add::format(f, &it.lhs, &it.rhs, &it.out),
            Instruction::Slice {
                input,
                start,
                end,
                out,
            } => {
                let item = out.item();
                f.write_fmt(format_args!("uint {out}_length = {end} - {start};\n"))?;
                f.write_fmt(format_args!("{item} *{out} = {input} + {start};\n"))
            }
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
            Instruction::SliceLength { input, out } => {
                f.write_fmt(format_args!("{out} = {input}_length;\n"))
            }
            Instruction::Length {
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
                let factor = match input.item() {
                    super::Item::Vec4(_) => 4,
                    super::Item::Vec3(_) => 3,
                    super::Item::Vec2(_) => 2,
                    super::Item::Scalar(_) => {
                        return f.write_fmt(format_args!(
                            "{out} = info[({offset} * 2 * info[0]) + {index}];\n"
                        ))
                    }
                };
                f.write_fmt(format_args!(
                    "{out} = info[({offset} * 2 * info[0]) + {index}] / {factor};\n"
                ))
            }
            Instruction::Wrap(it) => f.write_fmt(format_args!("{it}")),
            Instruction::Fma { a, b, c, out } => Fma::format(f, a, b, c, out),
            Instruction::Wmma(it) => f.write_fmt(format_args!("{it}")),
        }
    }
}

struct Fma;

impl Fma {
    fn format(
        f: &mut core::fmt::Formatter<'_>,
        a: &Variable,
        b: &Variable,
        c: &Variable,
        out: &Variable,
    ) -> core::fmt::Result {
        let num = match out.item() {
            super::Item::Vec4(_) => 4,
            super::Item::Vec3(_) => 3,
            super::Item::Vec2(_) => 2,
            super::Item::Scalar(_) => 1,
        };

        for i in 0..num {
            let ai = a.index(i);
            let bi = b.index(i);
            let ci = c.index(i);
            let outi = out.index(i);

            f.write_fmt(format_args!("{outi} = fma({ai}, {bi}, {ci});\n"))?;
        }

        Ok(())
    }
}
