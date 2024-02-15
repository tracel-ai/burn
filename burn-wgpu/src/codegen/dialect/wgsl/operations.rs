use super::base::{Item, Variable};
use std::fmt::Display;

/// All instructions that can be used in a WGSL compute shader.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Instruction {
    DeclareVariable {
        var: Variable,
    },
    Add {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Index {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Modulo {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sub {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Mul {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Div {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Abs {
        input: Variable,
        out: Variable,
    },
    Exp {
        input: Variable,
        out: Variable,
    },
    Log {
        input: Variable,
        out: Variable,
    },
    Log1p {
        input: Variable,
        out: Variable,
    },
    Cos {
        input: Variable,
        out: Variable,
    },
    Sin {
        input: Variable,
        out: Variable,
    },
    Tanh {
        input: Variable,
        out: Variable,
    },
    Powf {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sqrt {
        input: Variable,
        out: Variable,
    },
    Erf {
        input: Variable,
        out: Variable,
    },
    Recip {
        input: Variable,
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
    Clamp {
        input: Variable,
        min_value: Variable,
        max_value: Variable,
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
    ConditionalAssign {
        cond: Variable,
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AssignGlobal {
        input: Variable,
        out: Variable,
    },
    AssignLocal {
        input: Variable,
        out: Variable,
    },
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
    RangeLoop {
        i: Variable,
        start: Variable,
        end: Variable,
        instructions: Vec<Instruction>,
    },
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::DeclareVariable { var } => {
                let item = var.item();
                f.write_fmt(format_args!("var {var}: {item};\n"))
            }
            Instruction::Add { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} + {rhs};\n"))
            }
            Instruction::Index { lhs, rhs, out } => {
                let item = out.item();
                let lhs = match lhs {
                    Variable::GlobalInputArray(index, _) => format!("input_{index}_global"),
                    Variable::GlobalOutputArray(index, _) => format!("output_{index}_global"),
                    _ => format!("{lhs}"),
                };
                f.write_fmt(format_args!("{out} = {item}({lhs}[{rhs}]);\n"))
            }
            Instruction::Modulo { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} % {rhs};\n"))
            }
            Instruction::Sub { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} - {rhs};\n"))
            }
            Instruction::Mul { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} * {rhs};\n"))
            }
            Instruction::Div { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = {lhs} / {rhs};\n"))
            }
            Instruction::Abs { input, out } => f.write_fmt(format_args!("{out} = abs({input});\n")),
            Instruction::Exp { input, out } => f.write_fmt(format_args!("{out} = exp({input});\n")),
            Instruction::Log { input, out } => f.write_fmt(format_args!("{out} = log({input});\n")),
            Instruction::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => f.write_fmt(format_args!(
                "{out} = clamp({input}, {min_value}, {max_value});\n"
            )),
            Instruction::Powf { lhs, rhs, out } => {
                if rhs.is_always_scalar() {
                    f.write_fmt(format_args!("{out} = powf_scalar({lhs}, {rhs});\n"))
                } else {
                    f.write_fmt(format_args!("{out} = powf({lhs}, {rhs});\n"))
                }
            }
            Instruction::Sqrt { input, out } => {
                f.write_fmt(format_args!("{out} = sqrt({input});\n"))
            }
            Instruction::Log1p { input, out } => {
                f.write_fmt(format_args!("{out} = log({input} + 1.0);\n"))
            }
            Instruction::Cos { input, out } => f.write_fmt(format_args!("{out} = cos({input});\n")),
            Instruction::Sin { input, out } => f.write_fmt(format_args!("{out} = sin({input});\n")),
            Instruction::Tanh { input, out } => {
                #[cfg(target_os = "macos")]
                let result = f.write_fmt(format_args!("{out} = safe_tanh({input});\n"));
                #[cfg(not(target_os = "macos"))]
                let result = f.write_fmt(format_args!("{out} = tanh({input});\n"));

                result
            }
            Instruction::Erf { input, out } => f.write_fmt(format_args!("{out} = erf({input});\n")),
            Instruction::Recip { input, out } => {
                f.write_fmt(format_args!("{out} = 1.0 / {input};"))
            }
            Instruction::Equal { lhs, rhs, out } => comparison(lhs, rhs, out, "==", f),
            Instruction::Lower { lhs, rhs, out } => comparison(lhs, rhs, out, "<", f),
            Instruction::Greater { lhs, rhs, out } => comparison(lhs, rhs, out, ">", f),
            Instruction::LowerEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "<=", f),
            Instruction::GreaterEqual { lhs, rhs, out } => comparison(lhs, rhs, out, ">=", f),
            Instruction::AssignGlobal { input, out } => {
                let elem_out = out.item();
                let elem_in = input.item();

                if elem_out != elem_in {
                    match elem_out {
                        Item::Vec4(elem) => f.write_fmt(format_args!(
                            "
{out}_global[id] = vec4(
    {elem}({input}[0]),
    {elem}({input}[1]),
    {elem}({input}[2]),
    {elem}({input}[3]),
);"
                        )),
                        Item::Vec3(elem) => f.write_fmt(format_args!(
                            "
{out}_global[id] = vec3(
    {elem}({input}[0]),
    {elem}({input}[1]),
    {elem}({input}[2]),
);"
                        )),
                        Item::Vec2(elem) => f.write_fmt(format_args!(
                            "
{out}_global[id] = vec2(
    {elem}({input}[0]),
    {elem}({input}[1]),
);"
                        )),
                        Item::Scalar(elem) => {
                            f.write_fmt(format_args!("{out}_global[id] = {elem}({input});\n"))
                        }
                    }
                } else {
                    f.write_fmt(format_args!("{out}_global[id] = {elem_out}({input});\n"))
                }
            }
            Instruction::AssignLocal { input, out } => {
                let item = out.item();
                f.write_fmt(format_args!("{out} = {item}({input});\n"))
            }
            Instruction::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => {
                let elem = out.item();

                match elem {
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
if {cond}[0] {{
    {out}[0] = {lhs0};
}} else {{
    {out}[0] = {rhs0};
}}
if {cond}[1] {{
    {out}[1] = {lhs1};
}} else {{
    {out}[1] = {rhs1};
}}
if {cond}[2] {{
    {out}[2] = {lhs2};
}} else {{
    {out}[2] = {rhs2};
}}
if {cond}[3] {{
    {out}[3] = {lhs3};
}} else {{
    {out}[3] = {rhs3};
}}
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
if {cond}[0] {{
    {out}[0] = {lhs0};
}} else {{
    {out}[0] = {rhs0};
}}
if {cond}[1] {{
    {out}[1] = {lhs1};
}} else {{
    {out}[1] = {rhs1};
}}
if {cond}[2] {{
    {out}[2] = {lhs2};
}} else {{
    {out}[2] = {rhs2};
}}
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
if {cond}[0] {{
    {out}[0] = {lhs0};
}} else {{
    {out}[0] = {rhs0};
}}
if {cond}[1] {{
    {out}[1] = {lhs1};
}} else {{
    {out}[1] = {rhs1};
}}
"
                        ))
                    }
                    Item::Scalar(_) => f.write_fmt(format_args!(
                        "
if {cond} {{
    {out} = {lhs};
}} else {{
    {out} = {rhs};
}}
"
                    )),
                }
            }
            Instruction::Stride { dim, position, out } => f.write_fmt(format_args!(
                "{out} = info[({position}u * (2u * rank)) + {dim} + 1u];\n"
            )),
            Instruction::Shape { dim, position, out } => f.write_fmt(format_args!(
                "{out} = info[({position}u * (2u * rank)) + rank + {dim} + 1u];\n"
            )),
            Instruction::RangeLoop {
                i,
                start,
                end,
                instructions,
            } => {
                f.write_fmt(format_args!(
                    "
for (var {i}: u32 = {start}; {i} < {end}; {i}++) {{
"
                ))?;
                for instruction in instructions {
                    f.write_fmt(format_args!("{instruction}"))?;
                }

                f.write_str("}\n")
            }
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
