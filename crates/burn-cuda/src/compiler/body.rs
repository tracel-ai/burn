use super::Instruction;
use std::fmt::Display;

/// A body is composed of a list of [operations](Operation).
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
                "
    const uint WORKGROUP_X = 32;
    uint globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint globalIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint id = globalIdx_y * (blockDim.x * WORKGROUP_X) + globalIdx_x;
",
            )?;
        }
        if self.rank || self.stride || self.shape {
            f.write_str("uint rank = info[0];\n")?;
        }

        if self.stride || self.shape {
            f.write_str("uint rank_2 = rank * 2;\n")?;
        }

        for ops in self.instructions.iter() {
            f.write_fmt(format_args!("{ops}"))?;
        }

        Ok(())
    }
}
