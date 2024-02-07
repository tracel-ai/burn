use crate::codegen::{wgsl::operations::WgslOperation, Body};
use std::fmt::Display;

/// A body is composed of a list of [operators](Operator).
///
/// Note that the body assumes that the kernel will run on a 2D grid defined by the workgroup size
/// X and Y, but with Z=1.
#[derive(Debug, Clone)]
pub struct WgslBody {
    pub operators: Vec<WgslOperation>,
}

impl Display for WgslBody {
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

impl From<Body> for WgslBody {
    fn from(value: Body) -> Self {
        Self {
            operators: value.operators.into_iter().map(From::from).collect(),
        }
    }
}
