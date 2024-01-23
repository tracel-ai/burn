use super::{variable::Variable, Item, Vectorization};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// All operators that can be fused in a WGSL compute shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Operator {
    Add {
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
    ReadGlobal {
        variable: Variable,
    },
    /// Read the tensor in a way to be compatible with another tensor layout.
    ReadGlobalWithLayout {
        variable: Variable,
        tensor_read_pos: usize,
        tensor_layout_pos: usize,
    },
}

impl Operator {
    pub fn vectorize(&self, vectorize: Vectorization) -> Self {
        match self {
            Operator::Add { lhs, rhs, out } => Operator::Add {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Sub { lhs, rhs, out } => Operator::Sub {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Mul { lhs, rhs, out } => Operator::Mul {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Div { lhs, rhs, out } => Operator::Div {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Abs { input, out } => Operator::Abs {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Tanh { input, out } => Operator::Tanh {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Sin { input, out } => Operator::Sin {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Cos { input, out } => Operator::Cos {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Log1p { input, out } => Operator::Log1p {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Log { input, out } => Operator::Log {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Exp { input, out } => Operator::Exp {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Sqrt { input, out } => Operator::Sqrt {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Erf { input, out } => Operator::Erf {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Powf { lhs, rhs, out } => Operator::Powf {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Equal { lhs, rhs, out } => Operator::Equal {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Lower { lhs, rhs, out } => Operator::Lower {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Greater { lhs, rhs, out } => Operator::Greater {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::LowerEqual { lhs, rhs, out } => Operator::LowerEqual {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::GreaterEqual { lhs, rhs, out } => Operator::GreaterEqual {
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Recip { input, out } => Operator::Recip {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::AssignGlobal { input, out } => Operator::AssignGlobal {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::AssignLocal { input, out } => Operator::AssignLocal {
                input: input.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => Operator::Clamp {
                input: input.vectorize(vectorize),
                min_value: min_value.vectorize(vectorize),
                max_value: max_value.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => Operator::ConditionalAssign {
                cond: cond.vectorize(vectorize),
                lhs: lhs.vectorize(vectorize),
                rhs: rhs.vectorize(vectorize),
                out: out.vectorize(vectorize),
            },
            Operator::ReadGlobal { variable } => Operator::ReadGlobal {
                variable: variable.vectorize(vectorize),
            },
            Operator::ReadGlobalWithLayout {
                variable,
                tensor_read_pos,
                tensor_layout_pos,
            } => Operator::ReadGlobalWithLayout {
                variable: variable.vectorize(vectorize),
                tensor_read_pos: *tensor_read_pos,
                tensor_layout_pos: *tensor_layout_pos,
            },
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} + {rhs};"))
            }
            Operator::Sub { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} - {rhs};"))
            }
            Operator::Mul { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} * {rhs};"))
            }
            Operator::Div { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} / {rhs};"))
            }
            Operator::Abs { input, out } => f.write_fmt(format_args!("let {out} = abs({input});")),
            Operator::Exp { input, out } => f.write_fmt(format_args!("let {out} = exp({input});")),
            Operator::Log { input, out } => f.write_fmt(format_args!("let {out} = log({input});")),
            Operator::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => f.write_fmt(format_args!(
                "let {out} = clamp({input}, {min_value}, {max_value});"
            )),
            Operator::Powf { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = powf({lhs}, {rhs});"))
            }
            Operator::Sqrt { input, out } => {
                f.write_fmt(format_args!("let {out} = sqrt({input});"))
            }
            Operator::Log1p { input, out } => {
                f.write_fmt(format_args!("let {out} = log({input} + 1.0);"))
            }
            Operator::Cos { input, out } => f.write_fmt(format_args!("let {out} = cos({input});")),
            Operator::Sin { input, out } => f.write_fmt(format_args!("let {out} = sin({input});")),
            Operator::Tanh { input, out } => {
                #[cfg(target_os = "macos")]
                let result = f.write_fmt(format_args!("let {out} = safe_tanh({input});"));
                #[cfg(not(target_os = "macos"))]
                let result = f.write_fmt(format_args!("let {out} = tanh({input});"));

                result
            }
            Operator::Erf { input, out } => f.write_fmt(format_args!("let {out} = erf({input});")),
            Operator::Recip { input, out } => {
                f.write_fmt(format_args!("let {out} = 1.0 / {input};"))
            }
            Operator::Equal { lhs, rhs, out } => comparison(lhs, rhs, out, "==", f),
            Operator::Lower { lhs, rhs, out } => comparison(lhs, rhs, out, "<", f),
            Operator::Greater { lhs, rhs, out } => comparison(lhs, rhs, out, ">", f),
            Operator::LowerEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "<=", f),
            Operator::GreaterEqual { lhs, rhs, out } => comparison(lhs, rhs, out, ">=", f),
            Operator::AssignGlobal { input, out } => {
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
                            f.write_fmt(format_args!("{out}_global[id] = {elem}({input});"))
                        }
                    }
                } else {
                    f.write_fmt(format_args!("{out}_global[id] = {elem_out}({input});"))
                }
            }
            Operator::AssignLocal { input, out } => {
                let elem = out.item();
                f.write_fmt(format_args!("let {out} = {elem}({input});"))
            }
            Operator::ReadGlobal { variable } => match variable {
                Variable::Input(number, _elem) => f.write_fmt(format_args!(
                    "let input_{number} = input_{number}_global[id];"
                )),
                Variable::Local(_, _) => panic!("can't read global local variable."),
                Variable::Output(number, _elem) => f.write_fmt(format_args!(
                    "let output_{number} = output_{number}_global[id];"
                )),
                Variable::Scalar(_, _) => panic!("Can't read global scalar variable."),
            },
            Operator::ReadGlobalWithLayout {
                variable,
                tensor_read_pos: position,
                tensor_layout_pos: position_out,
            } => {
                let (global, local, elem) = match variable {
                    Variable::Input(number, elem) => (
                        format!("input_{number}_global"),
                        format!("input_{number}"),
                        elem,
                    ),
                    Variable::Local(_, _) => panic!("can't read global local variable."),
                    Variable::Output(number, elem) => (
                        format!("output_{number}_global"),
                        format!("output_{number}"),
                        elem,
                    ),
                    Variable::Scalar(_, _) => panic!("Can't read global scalar variable."),
                };

                let offset = match elem {
                    super::Item::Vec4(_) => 4,
                    super::Item::Vec3(_) => 3,
                    super::Item::Vec2(_) => 2,
                    super::Item::Scalar(_) => 1,
                };

                f.write_fmt(format_args!(
                    "
var index_{local}: u32 = 0u;

for (var i: u32 = 1u; i <= rank; i++) {{
    let position = {position}u * (2u * rank);
    let position_out = {position_out}u * (2u * rank);

    let stride = info[position + i];
    let stride_out = info[position_out + i];
    let shape = info[position + rank + i];

    index_{local} += (id * {offset}u) / stride_out % shape * stride;
}}

let {local} = {elem}({global}[index_{local} /  {offset}u]);
"
                ))
            }
            Operator::ConditionalAssign {
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
var {out}: {elem};
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
var {out}: {elem};
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
var {out}: {elem};
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
var {out}: {elem};
if {cond} {{
    {out} = {lhs};
}} else {{
    {out} = {rhs};
}}
"
                    )),
                }
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
let {out} = vec4({lhs0} {op} {rhs0}, {lhs1} {op} {rhs1}, {lhs2} {op} {rhs2}, {lhs3} {op} {rhs3});
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
let {out} = vec3({lhs0} {op} {rhs0}, {lhs1} {op} {rhs1}, {lhs2} {op} {rhs2});
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
let {out} = vec2({lhs0} {op} {rhs0}, {lhs1} {op} {rhs1});
"
            ))
        }
        Item::Scalar(_) => match rhs.item() {
            Item::Scalar(_) => f.write_fmt(format_args!("let {out} = {lhs} {op} {rhs};")),
            _ => panic!("Can only compare a scalar when the output is a scalar"),
        },
    }
}
