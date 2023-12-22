use super::variable::Variable;
use std::fmt::Display;

/// All operators that can be fused in a WGSL compute shader.
#[derive(Debug, Clone)]
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
            Operator::Equal { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} == {rhs};"))
            }
            Operator::Lower { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} < {rhs};"))
            }
            Operator::Greater { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} > {rhs};"))
            }
            Operator::LowerEqual { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} <= {rhs};"))
            }
            Operator::GreaterEqual { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} >= {rhs};"))
            }
            Operator::AssignGlobal { input, out } => {
                let elem = out.elem();
                f.write_fmt(format_args!("{out}_global[id] = {elem}({input});"))
            }
            Operator::AssignLocal { input, out } => {
                let elem = out.elem();
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

                f.write_fmt(format_args!(
                    "
var index_{local}: u32 = 0u;

for (var i: u32 = 1u; i <= rank; i++) {{
    let position = {position}u * (2u * rank);
    let position_out = {position_out}u * (2u * rank);

    let stride = info[position + i];
    let stride_out = info[position_out + i];
    let shape = info[position + rank + i];

    index_{local} += id / stride_out % shape * stride;
}}

let {local} = {elem}({global}[index_{local}]);
"
                ))
            }
            Operator::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => {
                let elem = out.elem();
                f.write_fmt(format_args!(
                    "
var {out}: {elem};
if {cond} {{
    {out} = {lhs};
}} else {{
    {out} = {rhs};
}}
"
                ))
            }
        }
    }
}
