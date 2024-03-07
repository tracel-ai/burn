use std::fmt::Display;

use super::{Item, Variable};

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
            Instruction::Add { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} + {rhs};\n"))
            }
            Instruction::Modulo { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} % {rhs};\n"))
            }

            Instruction::Mul { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} * {rhs};\n"))
            }

            Instruction::Div { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} / {rhs};\n"))
            }
            Instruction::Sub { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} - {rhs};\n"))
            }
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
            Instruction::Assign { input, out } => match out.item() {
                Item::Vec4(elem) => {
                    let input0 = input.index(0);
                    let input1 = input.index(1);
                    let input2 = input.index(2);
                    let input3 = input.index(3);

                    f.write_fmt(format_args!(
                        "{out} = vec4(
    {elem}({input0}),
    {elem}({input1}),
    {elem}({input2}),
    {elem}({input3}),
);
"
                    ))
                }
                Item::Vec3(elem) => {
                    let input0 = input.index(0);
                    let input1 = input.index(1);
                    let input2 = input.index(2);

                    f.write_fmt(format_args!(
                        "{out} = vec3(
    {elem}({input0}),
    {elem}({input1}),
    {elem}({input2}),
);
"
                    ))
                }
                Item::Vec2(elem) => {
                    let input0 = input.index(0);
                    let input1 = input.index(1);

                    f.write_fmt(format_args!(
                        "{out} = vec2(
    {elem}({input0}),
    {elem}({input1}),
);
"
                    ))
                }
                Item::Scalar(elem) => f.write_fmt(format_args!("{out} = {elem}({input});\n")),
            },
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
            Instruction::Equal { lhs, rhs, out } => comparison(lhs, rhs, out, "==", f),
            Instruction::Lower { lhs, rhs, out } => comparison(lhs, rhs, out, "<", f),
            Instruction::Greater { lhs, rhs, out } => comparison(lhs, rhs, out, ">", f),
            Instruction::LowerEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "<=", f),
            Instruction::GreaterEqual { lhs, rhs, out } => comparison(lhs, rhs, out, ">=", f),
        }
    }
}

fn comparison(
    lhs: &Variable,
    rhs: &Variable,
    out: &Variable,
    op: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    match out.item() {
        Item::Vec4(_) => {
            let lhs0 = lhs.index(0);
            let lhs1 = lhs.index(1);
            let lhs2 = lhs.index(2);
            let lhs3 = lhs.index(3);
            let rhs0 = rhs.index(0);
            let rhs1 = rhs.index(1);
            let rhs2 = rhs.index(2);
            let rhs3 = rhs.index(3);

            f.write_fmt(format_args!(
                "
{out} = vec4({lhs0} {op} {rhs0}, {lhs1} {op} {rhs1}, {lhs2} {op} {rhs2}, {lhs3} {op} {rhs3});
"
            ))
        }
        Item::Vec3(_) => {
            let lhs0 = lhs.index(0);
            let lhs1 = lhs.index(1);
            let lhs2 = lhs.index(2);
            let rhs0 = rhs.index(0);
            let rhs1 = rhs.index(1);
            let rhs2 = rhs.index(2);

            f.write_fmt(format_args!(
                "
{out} = vec3({lhs0} {op} {rhs0}, {lhs1} {op} {rhs1}, {lhs2} {op} {rhs2});
"
            ))
        }
        Item::Vec2(_) => {
            let lhs0 = lhs.index(0);
            let lhs1 = lhs.index(1);
            let rhs0 = rhs.index(0);
            let rhs1 = rhs.index(1);

            f.write_fmt(format_args!(
                "
{out} = vec2({lhs0} {op} {rhs0}, {lhs1} {op} {rhs1});
"
            ))
        }
        Item::Scalar(_) => match rhs.item() {
            Item::Scalar(_) => f.write_fmt(format_args!("{out} = {lhs} {op} {rhs};\n")),
            _ => panic!("Can only compare a scalar when the output is a scalar"),
        },
    }
}
