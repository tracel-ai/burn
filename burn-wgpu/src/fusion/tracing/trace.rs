use super::{ExecutionInfo, Scalars};
use crate::codegen::{dialect::gpu, CompilationInfo, InplaceMapping, Input, Output};
use burn_fusion::TensorDescription;
use serde::{Deserialize, Serialize};

#[derive(new, Clone, Serialize, Deserialize)]
pub struct Trace {
    inputs: Vec<(TensorDescription, gpu::Elem)>,
    outputs: Vec<(TensorDescription, gpu::Elem)>,
    locals: Vec<u16>,
    scalars: Scalars,
    scope: gpu::Scope,
}

impl Trace {
    pub fn running(&self) -> ExecutionInfo<'_> {
        ExecutionInfo {
            inputs: self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            outputs: self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            scalars: &self.scalars,
        }
    }

    pub fn compiling(&self) -> CompilationInfo {
        let mut inputs = self
            .inputs
            .iter()
            .map(|(_tensor, elem)| Input::Array {
                item: gpu::Item::Scalar(*elem),
                visibility: gpu::Visibility::Read,
            })
            .collect::<Vec<_>>();

        let outputs = self
            .outputs
            .iter()
            .zip(self.locals.iter())
            .map(|((_tensor, elem), local)| Output::Array {
                item: gpu::Item::Scalar(*elem),
                local: *local,
            })
            .collect::<Vec<_>>();

        if self.scalars.num_float > 0 {
            inputs.push(Input::Scalar {
                elem: gpu::Elem::Float,
                size: self.scalars.num_float,
            })
        }

        if self.scalars.num_uint > 0 {
            inputs.push(Input::Scalar {
                elem: gpu::Elem::UInt,
                size: self.scalars.num_uint,
            })
        }

        if self.scalars.num_int > 0 {
            inputs.push(Input::Scalar {
                elem: gpu::Elem::Int,
                size: self.scalars.num_int,
            })
        }

        let mut potential_inplace = self
            .inputs
            .iter()
            .zip(inputs.iter())
            .enumerate()
            .filter(|(_pos, ((desc, _elem), _input))| match desc.status {
                burn_fusion::TensorStatus::ReadOnly => false,
                burn_fusion::TensorStatus::ReadWrite => true,
                burn_fusion::TensorStatus::NotInit => false,
            })
            .map(|(pos, ((desc, elem), input))| (pos, desc, elem, input))
            .collect::<Vec<_>>();

        let mappings = self
            .outputs
            .iter()
            .zip(outputs.iter())
            .enumerate()
            .filter_map(|(pos, ((desc, elem), _output))| {
                if potential_inplace.is_empty() {
                    return None;
                }

                let mut chosen = None;
                for (index, (_pos_input, desc_input, elem_input, _input)) in
                    potential_inplace.iter().enumerate()
                {
                    if chosen.is_some() {
                        break;
                    }
                    if desc.shape == desc_input.shape && *elem_input == elem {
                        chosen = Some(index);
                    }
                }

                match chosen {
                    Some(index) => {
                        let input = potential_inplace.remove(index);
                        Some(InplaceMapping::new(input.0, pos))
                    }
                    None => None,
                }
            })
            .collect::<Vec<_>>();

        CompilationInfo {
            inputs,
            outputs,
            scope: self.scope.clone(),
            mappings,
        }
    }
}
