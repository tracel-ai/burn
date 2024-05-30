use super::Variable;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum Subgroup {
    SubgroupElect {
        out: Variable,
    },
    SubgroupAll {
        input: Variable,
        out: Variable,
    },
    SubgroupAny {
        input: Variable,
        out: Variable,
    },
    SubgroupBroadcast {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    SubgroupSum {
        input: Variable,
        out: Variable,
    },
    SubgroupProduct {
        input: Variable,
        out: Variable,
    },
    SubgroupAnd {
        input: Variable,
        out: Variable,
    },
    SubgroupOr {
        input: Variable,
        out: Variable,
    },
    SubgroupXor {
        input: Variable,
        out: Variable,
    },
    SubgroupMin {
        input: Variable,
        out: Variable,
    },
    SubgroupMax {
        input: Variable,
        out: Variable,
    },
}

impl Display for Subgroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Subgroup::SubgroupElect { out } => {
                f.write_fmt(format_args!("{out} = subgroupElect();\n"))
            }
            Subgroup::SubgroupAll { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAll({input});\n"))
            }
            Subgroup::SubgroupAny { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAny({input});\n"))
            }
            Subgroup::SubgroupBroadcast { lhs, rhs, out } => {
                f.write_fmt(format_args!("{out} = subgroupBroadcast({lhs}, {rhs});\n"))
            }
            Subgroup::SubgroupSum { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAdd({input});\n"))
            }
            Subgroup::SubgroupProduct { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupMul({input});\n"))
            }
            Subgroup::SubgroupAnd { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupAnd({input});\n"))
            }
            Subgroup::SubgroupOr { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupOr({input});\n"))
            }
            Subgroup::SubgroupXor { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupXor({input});\n"))
            }
            Subgroup::SubgroupMin { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupMin({input});\n"))
            }
            Subgroup::SubgroupMax { input, out } => {
                f.write_fmt(format_args!("{out} = subgroupMax({input});\n"))
            }
        }
    }
}
