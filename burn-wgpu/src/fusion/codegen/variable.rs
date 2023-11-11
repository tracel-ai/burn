use std::fmt::Display;

#[derive(Debug, Hash, Clone)]
pub enum Variable {
    Input(u16),
    Local(u16),
    Output(u16),
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Input(number) => f.write_fmt(format_args!("input_{number}")),
            Variable::Local(number) => f.write_fmt(format_args!("local_{number}")),
            Variable::Output(number) => f.write_fmt(format_args!("output_{number}")),
        }
    }
}
