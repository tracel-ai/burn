use super::Variable;
use std::fmt::Display;

#[derive(Debug, Hash, Clone)]
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
    Exp {
        input: Variable,
        out: Variable,
    },
    AssignGlobal {
        input: Variable,
        out: Variable,
    },
    ReadGlobal {
        variable: Variable,
        position: usize,
        position_out: usize,
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
            Operator::Exp { input, out } => f.write_fmt(format_args!("let {out} = exp({input});")),
            Operator::AssignGlobal { input, out } => {
                f.write_fmt(format_args!("{out}_global[id] = {input};"))
            }
            Operator::ReadGlobal {
                variable,
                position,
                position_out,
            } => {
                let (global, local) = match variable {
                    Variable::Input(number) => {
                        (format!("input_{number}_global"), format!("input_{number}"))
                    }
                    Variable::Local(_) => panic!("can't ready global a temp variable."),
                    Variable::Output(number) => (
                        format!("output_{number}_global"),
                        format!("output_{number}"),
                    ),
                };

                f.write_fmt(format_args!(
                    "
var index_{local}: u32 = 0u;

for (var i: u32 = 1u; i <= dim; i++) {{
    let position = {position}u * (2u * dim);
    let position_out = {position_out}u * (2u * dim);

    let stride = info[position + i];
    let stride_out = info[position_out + i];
    let shape = info[position + dim + i];

    index_{local} += id / stride_out % shape * stride;
}}

let {local} = {global}[index_{local}];
"
                ))
            }
        }
    }
}
