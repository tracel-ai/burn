use super::{binary::*, unary::*, Component, Item, Variable};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Instruction {
    DeclareVariable {
        var: Variable,
    },
    Modulo {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Add {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Div {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Mul {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sub {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Index {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    IndexAssign {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Assign {
        input: Variable,
        out: Variable,
    },
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
    Equal {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Lower {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Greater {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    LowerEqual {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    GreaterEqual {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Erf {
        input: Variable,
        out: Variable,
    },
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
            Instruction::Add { lhs, rhs, out } => Add::format(f, lhs, rhs, out),
            Instruction::Mul { lhs, rhs, out } => Mul::format(f, lhs, rhs, out),
            Instruction::Div { lhs, rhs, out } => Div::format(f, lhs, rhs, out),
            Instruction::Sub { lhs, rhs, out } => Sub::format(f, lhs, rhs, out),
            Instruction::Modulo { lhs, rhs, out } => Modulo::format(f, lhs, rhs, out),
            Instruction::Index { lhs, rhs, out } => {
                let item = out.item();
                f.write_fmt(format_args!("{out} = {item}({lhs}[{rhs}]);\n"))
            }
            Instruction::IndexAssign { lhs, rhs, out } => match lhs.item() {
                Item::Vec4(elem) => {
                    let lhs0 = lhs.index(0);
                    let lhs1 = lhs.index(1);
                    let lhs2 = lhs.index(2);
                    let lhs3 = lhs.index(3);

                    let rhs0 = rhs.index(0);
                    let rhs1 = rhs.index(1);
                    let rhs2 = rhs.index(2);
                    let rhs3 = rhs.index(3);

                    f.write_fmt(format_args!("{out}[{lhs0}] = {elem}({rhs0});\n"))?;
                    f.write_fmt(format_args!("{out}[{lhs1}] = {elem}({rhs1});\n"))?;
                    f.write_fmt(format_args!("{out}[{lhs2}] = {elem}({rhs2});\n"))?;
                    f.write_fmt(format_args!("{out}[{lhs3}] = {elem}({rhs3});\n"))
                }
                Item::Vec3(elem) => {
                    let lhs0 = lhs.index(0);
                    let lhs1 = lhs.index(1);
                    let lhs2 = lhs.index(2);

                    let rhs0 = rhs.index(0);
                    let rhs1 = rhs.index(1);
                    let rhs2 = rhs.index(2);

                    f.write_fmt(format_args!("{out}[{lhs0}] = {elem}({rhs0});\n"))?;
                    f.write_fmt(format_args!("{out}[{lhs1}] = {elem}({rhs1});\n"))?;
                    f.write_fmt(format_args!("{out}[{lhs2}] = {elem}({rhs2});\n"))
                }
                Item::Vec2(elem) => {
                    let lhs0 = lhs.index(0);
                    let lhs1 = lhs.index(1);

                    let rhs0 = rhs.index(0);
                    let rhs1 = rhs.index(1);

                    f.write_fmt(format_args!("{out}[{lhs0}] = {elem}({rhs0});\n"))?;
                    f.write_fmt(format_args!("{out}[{lhs1}] = {elem}({rhs1});\n"))
                }
                Item::Scalar(_elem) => {
                    let elem_out = out.elem();
                    let casting_type = match rhs.item() {
                        Item::Vec4(_) => Item::Vec4(elem_out),
                        Item::Vec3(_) => Item::Vec3(elem_out),
                        Item::Vec2(_) => Item::Vec2(elem_out),
                        Item::Scalar(_) => Item::Scalar(elem_out),
                    };
                    f.write_fmt(format_args!("{out}[{lhs}] = {casting_type}({rhs});\n"))
                }
            },
            Instruction::Assign { input, out } => Assign::format(f, input, out),
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
            Instruction::Equal { lhs, rhs, out } => Equal::format(f, lhs, rhs, out),
            Instruction::Lower { lhs, rhs, out } => Lower::format(f, lhs, rhs, out),
            Instruction::Greater { lhs, rhs, out } => Greater::format(f, lhs, rhs, out),
            Instruction::LowerEqual { lhs, rhs, out } => LowerEqual::format(f, lhs, rhs, out),
            Instruction::GreaterEqual { lhs, rhs, out } => GreaterEqual::format(f, lhs, rhs, out),
            Instruction::Erf { input, out } => Erf::format(f, input, out),
        }
    }
}
