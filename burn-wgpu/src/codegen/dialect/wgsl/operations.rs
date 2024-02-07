use super::base::{WgslItem, WgslVariable};
use crate::codegen::dialect::gpu;
use std::fmt::Display;

/// All operators that can be fused in a WGSL compute shader.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum WgslOperation {
    Add {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Sub {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Mul {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Div {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Abs {
        input: WgslVariable,
        out: WgslVariable,
    },
    Exp {
        input: WgslVariable,
        out: WgslVariable,
    },
    Log {
        input: WgslVariable,
        out: WgslVariable,
    },
    Log1p {
        input: WgslVariable,
        out: WgslVariable,
    },
    Cos {
        input: WgslVariable,
        out: WgslVariable,
    },
    Sin {
        input: WgslVariable,
        out: WgslVariable,
    },
    Tanh {
        input: WgslVariable,
        out: WgslVariable,
    },
    Powf {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Sqrt {
        input: WgslVariable,
        out: WgslVariable,
    },
    Erf {
        input: WgslVariable,
        out: WgslVariable,
    },
    Recip {
        input: WgslVariable,
        out: WgslVariable,
    },
    Equal {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Lower {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    Clamp {
        input: WgslVariable,
        min_value: WgslVariable,
        max_value: WgslVariable,
        out: WgslVariable,
    },
    Greater {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    LowerEqual {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    GreaterEqual {
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    ConditionalAssign {
        cond: WgslVariable,
        lhs: WgslVariable,
        rhs: WgslVariable,
        out: WgslVariable,
    },
    AssignGlobal {
        input: WgslVariable,
        out: WgslVariable,
    },
    AssignLocal {
        input: WgslVariable,
        out: WgslVariable,
    },
    ReadGlobal {
        variable: WgslVariable,
    },
    /// Read the tensor in a way to be compatible with another tensor layout.
    ReadGlobalWithLayout {
        variable: WgslVariable,
        tensor_read_pos: usize,
        tensor_layout_pos: usize,
    },
}

impl From<gpu::Operation> for WgslOperation {
    fn from(value: gpu::Operation) -> Self {
        match value {
            gpu::Operation::Add(op) => Self::Add {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Sub(op) => Self::Sub {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Mul(op) => Self::Mul {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Div(op) => Self::Div {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Abs(op) => Self::Abs {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Exp(op) => Self::Exp {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Log(op) => Self::Log {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Log1p(op) => Self::Log1p {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Cos(op) => Self::Cos {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Sin(op) => Self::Sin {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Tanh(op) => Self::Tanh {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Powf(op) => Self::Powf {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Sqrt(op) => Self::Sqrt {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Erf(op) => Self::Erf {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Recip(op) => Self::Recip {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::Equal(op) => Self::Equal {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Lower(op) => Self::Lower {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::Clamp(op) => Self::Clamp {
                input: op.input.into(),
                min_value: op.min_value.into(),
                max_value: op.max_value.into(),
                out: op.out.into(),
            },
            gpu::Operation::Greater(op) => Self::Greater {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::LowerEqual(op) => Self::LowerEqual {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::GreaterEqual(op) => Self::GreaterEqual {
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::ConditionalAssign(op) => Self::ConditionalAssign {
                cond: op.cond.into(),
                lhs: op.lhs.into(),
                rhs: op.rhs.into(),
                out: op.out.into(),
            },
            gpu::Operation::AssignGlobal(op) => Self::AssignGlobal {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::AssignLocal(op) => Self::AssignLocal {
                input: op.input.into(),
                out: op.out.into(),
            },
            gpu::Operation::ReadGlobal(op) => Self::ReadGlobal {
                variable: op.variable.into(),
            },
            gpu::Operation::ReadGlobalWithLayout(op) => Self::ReadGlobalWithLayout {
                variable: op.variable.into(),
                tensor_read_pos: op.tensor_read_pos.into(),
                tensor_layout_pos: op.tensor_layout_pos.into(),
            },
        }
    }
}

impl Display for WgslOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslOperation::Add { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} + {rhs};"))
            }
            WgslOperation::Sub { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} - {rhs};"))
            }
            WgslOperation::Mul { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} * {rhs};"))
            }
            WgslOperation::Div { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} / {rhs};"))
            }
            WgslOperation::Abs { input, out } => {
                f.write_fmt(format_args!("let {out} = abs({input});"))
            }
            WgslOperation::Exp { input, out } => {
                f.write_fmt(format_args!("let {out} = exp({input});"))
            }
            WgslOperation::Log { input, out } => {
                f.write_fmt(format_args!("let {out} = log({input});"))
            }
            WgslOperation::Clamp {
                input,
                min_value,
                max_value,
                out,
            } => f.write_fmt(format_args!(
                "let {out} = clamp({input}, {min_value}, {max_value});"
            )),
            WgslOperation::Powf { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = powf({lhs}, {rhs});"))
            }
            WgslOperation::Sqrt { input, out } => {
                f.write_fmt(format_args!("let {out} = sqrt({input});"))
            }
            WgslOperation::Log1p { input, out } => {
                f.write_fmt(format_args!("let {out} = log({input} + 1.0);"))
            }
            WgslOperation::Cos { input, out } => {
                f.write_fmt(format_args!("let {out} = cos({input});"))
            }
            WgslOperation::Sin { input, out } => {
                f.write_fmt(format_args!("let {out} = sin({input});"))
            }
            WgslOperation::Tanh { input, out } => {
                #[cfg(target_os = "macos")]
                let result = f.write_fmt(format_args!("let {out} = safe_tanh({input});"));
                #[cfg(not(target_os = "macos"))]
                let result = f.write_fmt(format_args!("let {out} = tanh({input});"));

                result
            }
            WgslOperation::Erf { input, out } => {
                f.write_fmt(format_args!("let {out} = erf({input});"))
            }
            WgslOperation::Recip { input, out } => {
                f.write_fmt(format_args!("let {out} = 1.0 / {input};"))
            }
            WgslOperation::Equal { lhs, rhs, out } => comparison(lhs, rhs, out, "==", f),
            WgslOperation::Lower { lhs, rhs, out } => comparison(lhs, rhs, out, "<", f),
            WgslOperation::Greater { lhs, rhs, out } => comparison(lhs, rhs, out, ">", f),
            WgslOperation::LowerEqual { lhs, rhs, out } => comparison(lhs, rhs, out, "<=", f),
            WgslOperation::GreaterEqual { lhs, rhs, out } => comparison(lhs, rhs, out, ">=", f),
            WgslOperation::AssignGlobal { input, out } => {
                let elem_out = out.item();
                let elem_in = input.item();

                if elem_out != elem_in {
                    match elem_out {
                        WgslItem::Vec4(elem) => f.write_fmt(format_args!(
                            "
{out}_global[id] = vec4(
    {elem}({input}[0]),
    {elem}({input}[1]),
    {elem}({input}[2]),
    {elem}({input}[3]),
);"
                        )),
                        WgslItem::Vec3(elem) => f.write_fmt(format_args!(
                            "
{out}_global[id] = vec3(
    {elem}({input}[0]),
    {elem}({input}[1]),
    {elem}({input}[2]),
);"
                        )),
                        WgslItem::Vec2(elem) => f.write_fmt(format_args!(
                            "
{out}_global[id] = vec2(
    {elem}({input}[0]),
    {elem}({input}[1]),
);"
                        )),
                        WgslItem::Scalar(elem) => {
                            f.write_fmt(format_args!("{out}_global[id] = {elem}({input});"))
                        }
                    }
                } else {
                    f.write_fmt(format_args!("{out}_global[id] = {elem_out}({input});"))
                }
            }
            WgslOperation::AssignLocal { input, out } => {
                let elem = out.item();
                f.write_fmt(format_args!("let {out} = {elem}({input});"))
            }
            WgslOperation::ReadGlobal { variable } => match variable {
                WgslVariable::Input(number, _elem) => f.write_fmt(format_args!(
                    "let input_{number} = input_{number}_global[id];"
                )),
                WgslVariable::Local(_, _) => panic!("can't read global local variable."),
                WgslVariable::Output(number, _elem) => f.write_fmt(format_args!(
                    "let output_{number} = output_{number}_global[id];"
                )),
                WgslVariable::Scalar(_, _) => panic!("Can't read global scalar variable."),
                WgslVariable::Constant(_, _) => panic!("Can't read global constant variable."),
            },
            WgslOperation::ReadGlobalWithLayout {
                variable,
                tensor_read_pos: position,
                tensor_layout_pos: position_out,
            } => {
                let (global, local, elem) = match variable {
                    WgslVariable::Input(number, elem) => (
                        format!("input_{number}_global"),
                        format!("input_{number}"),
                        elem,
                    ),
                    WgslVariable::Local(_, _) => panic!("can't read global local variable."),
                    WgslVariable::Output(number, elem) => (
                        format!("output_{number}_global"),
                        format!("output_{number}"),
                        elem,
                    ),
                    WgslVariable::Scalar(_, _) => panic!("Can't read global scalar variable."),
                    WgslVariable::Constant(_, _) => panic!("Can't read global constant variable."),
                };

                let offset = match elem {
                    WgslItem::Vec4(_) => 4,
                    WgslItem::Vec3(_) => 3,
                    WgslItem::Vec2(_) => 2,
                    WgslItem::Scalar(_) => 1,
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
            WgslOperation::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => {
                let elem = out.item();

                match elem {
                    WgslItem::Vec4(_) => {
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
                    WgslItem::Vec3(_) => {
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
                    WgslItem::Vec2(_) => {
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
                    WgslItem::Scalar(_) => f.write_fmt(format_args!(
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
    lhs: &WgslVariable,
    rhs: &WgslVariable,
    out: &WgslVariable,
    op: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    match out.item() {
        WgslItem::Vec4(_) => {
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
        WgslItem::Vec3(_) => {
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
        WgslItem::Vec2(_) => {
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
        WgslItem::Scalar(_) => match rhs.item() {
            WgslItem::Scalar(_) => f.write_fmt(format_args!("let {out} = {lhs} {op} {rhs};")),
            _ => panic!("Can only compare a scalar when the output is a scalar"),
        },
    }
}
