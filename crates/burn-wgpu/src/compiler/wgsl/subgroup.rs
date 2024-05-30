use super::Variable;
use std::fmt::Display;

#[derive(Debug, Clone)]
#[allow(dead_code, missing_docs)] // Some variants might not be used with different flags
pub enum Subgroup {
    Elect {
        out: Variable,
    },
    All {
        input: Variable,
        out: Variable,
    },
    Any {
        input: Variable,
        out: Variable,
    },
    Broadcast {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sum {
        input: Variable,
        out: Variable,
    },
    Prod {
        input: Variable,
        out: Variable,
    },
    And {
        input: Variable,
        out: Variable,
    },
    Or {
        input: Variable,
        out: Variable,
    },
    Xor {
        input: Variable,
        out: Variable,
    },
    Min {
        input: Variable,
        out: Variable,
    },
    Max {
        input: Variable,
        out: Variable,
    },
}

impl Display for Subgroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subgroup::Elect { out } => f.write_fmt(format_args!("{out} = subgroupElect();\n")),
            Subgroup::All { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAll({input});\n"))
            }
            Subgroup::Any { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAny({input});\n"))
            }
            Subgroup::Broadcast { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = subgroupBroadcast({lhs}, {rhs});\n"))
            }
            Subgroup::Sum { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAdd({input});\n"))
            }
            Subgroup::Prod { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupMul({input});\n"))
            }
            Subgroup::And { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAnd({input});\n"))
            }
            Subgroup::Or { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupOr({input});\n"))
            }
            Subgroup::Xor { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupXor({input});\n"))
            }
            Subgroup::Min { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupMin({input});\n"))
            }
            Subgroup::Max { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupMax({input});\n"))
            }
        }
    }
}
