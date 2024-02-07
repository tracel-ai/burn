use super::Operation;
use std::fmt::Display;

/// A body is composed of a list of [operations](Operation).
///
/// Note that the body assumes that the kernel will run on a 2D grid defined by the workgroup size
/// X and Y, but with Z=1.
#[derive(Debug, Clone)]
pub struct Body {
    pub operators: Vec<Operation>,
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;\n",
        )?;
        f.write_str("let rank: u32 = info[0];\n\n")?;

        for ops in self.operators.iter() {
            f.write_fmt(format_args!("{ops}"))?;
            f.write_str("\n")?;
        }

        Ok(())
    }
}
