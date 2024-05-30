use std::fmt::Display;

use super::Variable;

#[derive(Clone, Debug)]
pub enum WrapInstruction {
    ReduceSum { input: Variable, out: Variable },
    ReduceProd { input: Variable, out: Variable },
    ReduceMax { input: Variable, out: Variable },
    ReduceMin { input: Variable, out: Variable },
}

impl Display for WrapInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WrapInstruction::ReduceSum { input, out } => f.write_fmt(format_args!(
                "
{out} = {input};
                    {{
    for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
        {out} += __shfl_down_sync(0xFFFFFFFF, {out}, offset);
    }}
}}
                        "
            )),
            WrapInstruction::ReduceProd { input, out } => f.write_fmt(format_args!(
                "
{out} = {input};
                    {{
    for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
        {out} *= __shfl_down_sync(0xFFFFFFFF, {out}, offset);
    }}
}}
                        "
            )),
            WrapInstruction::ReduceMax { input, out } => f.write_fmt(format_args!(
                "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = max({out}, __shfl_down_sync(0xFFFFFFFF, {out}, offset));
}}
}}
                    "
            )),
            WrapInstruction::ReduceMin { input, out } => f.write_fmt(format_args!(
                "
{out} = {input};
                {{
for (int offset = warpSizeChecked / 2; offset > 0; offset /= 2) {{
    {out} = min({out}, __shfl_down_sync(0xFFFFFFFF, {out}, offset));
}}
}}
                    "
            )),
        }
    }
}
