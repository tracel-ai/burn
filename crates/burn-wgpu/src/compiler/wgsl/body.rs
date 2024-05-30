use super::Instruction;
use std::fmt::Display;

/// A body is composed of a list of [instructions](Instruction).
///
/// Note that the body assumes that the kernel will run on a 2D grid defined by the workgroup size
/// X and Y, but with Z=1.
#[derive(Debug, Clone)]
pub struct Body {
    pub instructions: Vec<Instruction>,
    pub rank: bool,
    pub id: bool,
    pub stride: bool,
    pub shape: bool,
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.id {
            f.write_str(
                "let id = (global_id.z * num_workgroups.x * WORKGROUP_SIZE_X * num_workgroups.y * WORKGROUP_SIZE_Y) + (global_id.y * num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;\n",
            )?;
        }
        if self.rank || self.stride || self.shape {
            f.write_str("let rank: u32 = info[0];\n")?;
        }

        if self.stride || self.shape {
            f.write_str("let rank_2: u32 = rank * 2u;\n")?;
        }

        for ops in self.instructions.iter() {
            f.write_fmt(format_args!("{ops}"))?;
        }

        Ok(())
    }
}
