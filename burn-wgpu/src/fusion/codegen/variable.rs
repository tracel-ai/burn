use super::Elem;
use std::fmt::Display;

#[derive(Debug, Hash, Clone)]
pub enum Variable {
    Input(u16, Elem),
    Scalar(u16, Elem),
    Local(u16, Elem),
    Output(u16, Elem),
}

impl Variable {
    pub fn elem(&self) -> &Elem {
        match self {
            Variable::Input(_, e) => e,
            Variable::Scalar(_, e) => e,
            Variable::Local(_, e) => e,
            Variable::Output(_, e) => e,
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Input(number, _) => f.write_fmt(format_args!("input_{number}")),
            Variable::Local(number, _) => f.write_fmt(format_args!("local_{number}")),
            Variable::Output(number, _) => f.write_fmt(format_args!("output_{number}")),
            Variable::Scalar(number, elem) => f.write_fmt(format_args!("scalars_{elem}[{number}]")),
        }
    }
}
