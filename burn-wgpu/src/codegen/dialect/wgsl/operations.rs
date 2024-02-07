use super::base::{Item, Variable};
use std::fmt::Display;

/// All operators that can be fused in a WGSL compute shader.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Operation {
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

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Add { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} + {rhs};"))
            }
            Operation::Sub { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} - {rhs};"))
            }
            Operation::Mul { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} * {rhs};"))
            }
            Operation::Div { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} / {rhs};"))
            }
            Operation::Abs { input, out } => f.write_fmt(format_args!("let {out} = abs({input});")),
            Operation::Exp { input, out } => f.write_fmt(format_args!("let {out} = exp({input});")),
            Operation::Log { input, out } => f.write_fmt(format_args!("let {out} = log({input});")),
            Operation::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => f.write_fmt(format_args!(
                "let {out} = clamp({input}, {min_value}, {max_value});"
            )),
            Operation::Powf { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = powf({lhs}, {rhs});"))
            }
            Operation::Sqrt { input, out } => {
                f.write_fmt(format_args!("let {out} = sqrt({input});"))
            }
            Operation::Log1p { input, out } => {
                f.write_fmt(format_args!("let {out} = log({input} + 1.0);"))
            }
            Operation::Cos { input, out } => f.write_fmt(format_args!("let {out} = cos({input});")),
            Operation::Sin { input, out } => f.write_fmt(format_args!("let {out} = sin({input});")),
            Operation::Tanh { input, out } => {
                #[cfg(target_os = "macos")]
                let result = f.write_fmt(format_args!("let {out} = safe_tanh({input});"));
                #[cfg(not(target_os = "macos"))]
                let result = f.write_fmt(format_args!("let {out} = tanh({input});"));

                result
            }
            Operation::Erf { input, out } => f.write_fmt(format_args!("let {out} = erf({input});")),
            Operation::Recip { input, out } => {
                f.write_fmt(format_args!("let {out} = 1.0 / {input};"))
            }
            Operation::Equal { lhs, rhs, out } => comparison(lhs, rhs, out, "==", f),
            Operation::Lower { lhs, rhs, out } => comparison(lhs, rhs, out, "<", f),
            Operation::Greater { lhs, rhs, out } => comparison(lhs, rhs, out, ">", f),
            Operation::LowerEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "<=", f),
            Operation::GreaterEqual { lhs, rhs, out } => comparison(lhs, rhs, out, ">=", f),
            Operation::AssignGlobal { input, out } => {
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
            Operation::AssignLocal { input, out } => {
                let elem = out.item();
                f.write_fmt(format_args!("let {out} = {elem}({input});"))
            }
            Operation::ReadGlobal { variable } => match variable {
                Variable::Input(number, _elem) => f.write_fmt(format_args!(
                    "let input_{number} = input_{number}_global[id];"
                )),
                Variable::Local(_, _) => panic!("can't read global local variable."),
                Variable::Output(number, _elem) => f.write_fmt(format_args!(
                    "let output_{number} = output_{number}_global[id];"
                )),
                Variable::Scalar(_, _, _) => panic!("Can't read global scalar variable."),
                Variable::Constant(_, _) => panic!("Can't read global constant variable."),
            },
            Operation::ReadGlobalWithLayout {
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
                    Variable::Scalar(_, _, _) => panic!("Can't read global scalar variable."),
                    Variable::Constant(_, _) => panic!("Can't read global constant variable."),
                };

                let offset = match elem {
                    Item::Vec4(_) => 4,
                    Item::Vec3(_) => 3,
                    Item::Vec2(_) => 2,
                    Item::Scalar(_) => 1,
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
            Operation::ConditionalAssign {
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
