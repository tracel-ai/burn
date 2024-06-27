use super::Instruction;
use std::fmt::Display;

/// A body is composed of a list of [instructions](Instruction).
#[derive(Debug, Clone)]
pub struct Body {
    pub instructions: Vec<Instruction>,
    pub shared_memories: Vec<super::SharedMemory>,
    pub local_arrays: Vec<super::LocalArray>,
    pub stride: bool,
    pub shape: bool,
    pub id: bool,
    pub rank: bool,
    pub invocation_index: bool,
    pub global_invocation_id: (bool, bool, bool),
    pub wrap_size_checked: bool,
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.id
            || self.global_invocation_id.0
            || self.global_invocation_id.1
            || self.global_invocation_id.2
        {
            f.write_str(
                "
    int3 globalInvocationId = make_int3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    );
",
            )?;
        }

        if self.id {
            f.write_str(
                "
    uint id = (globalInvocationId.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) + (globalInvocationId.y * gridDim.x * blockDim.x) + globalInvocationId.x;
",
            )?;
        }

        if self.invocation_index {
            f.write_str(
                "
    int invocationIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
            ",
            )?;
        }
        if self.wrap_size_checked {
            f.write_str(
                "
 int warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
",
            )?;
        }

        if self.rank || self.stride || self.shape {
            f.write_str("uint rank = info[0];\n")?;
        }

        if self.stride || self.shape {
            f.write_str("uint rank_2 = rank * 2;\n")?;
        }

        for shared in self.shared_memories.iter() {
            f.write_fmt(format_args!(
                "__shared__ {} shared_memory_{}[{}];\n",
                shared.item, shared.index, shared.size
            ))?;
        }

        // Local arrays
        for array in self.local_arrays.iter() {
            f.write_fmt(format_args!(
                "{} l_arr_{}_{}[{}];\n\n",
                array.item, array.index, array.depth, array.size
            ))?;
        }

        for ops in self.instructions.iter() {
            f.write_fmt(format_args!("{ops}"))?;
        }

        Ok(())
    }
}
