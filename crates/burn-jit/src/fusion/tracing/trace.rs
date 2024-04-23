use super::Scalars;
use crate::codegen::{dialect::gpu, CompilationInfo, InputInfo, OutputInfo};
use burn_tensor::repr::TensorDescription;
use serde::{Deserialize, Serialize};

/// A trace encapsulates all information necessary to perform the compilation and execution of
/// captured [tensor operations](burn_fusion::stream::OperationDescription).
///
/// A trace should be built using a [builder](super::TraceBuilder).
#[derive(new, Clone, Serialize, Deserialize)]
pub struct Trace {
    inputs: Vec<(TensorDescription, gpu::Elem)>,
    outputs: Vec<(TensorDescription, gpu::Elem)>,
    locals: Vec<u16>,
    scalars: Scalars,
    scope: gpu::Scope,
}

/// Information necessary to execute a kernel.
pub struct ExecutionInfo<'a> {
    /// Tensor inputs.
    pub inputs: Vec<&'a TensorDescription>,
    /// Tensor outputs.
    pub outputs: Vec<&'a TensorDescription>,
    /// Scalar inputs.
    pub scalars: &'a Scalars,
}

impl Trace {
    /// Collect information related to running the trace.
    pub fn running(&self) -> ExecutionInfo<'_> {
        ExecutionInfo {
            inputs: self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            outputs: self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            scalars: &self.scalars,
        }
    }

    /// Collect information related to compiling the trace.
    pub fn compiling(&self) -> CompilationInfo {
        let mut inputs = self
            .inputs
            .iter()
            .map(|(_tensor, elem)| InputInfo::Array {
                item: gpu::Item::Scalar(*elem),
                visibility: gpu::Visibility::Read,
            })
            .collect::<Vec<_>>();

        let outputs = self
            .outputs
            .iter()
            .zip(self.locals.iter())
            .map(|((_tensor, elem), local)| OutputInfo::ArrayWrite {
                item: gpu::Item::Scalar(*elem),
                local: *local,
            })
            .collect::<Vec<_>>();

        if self.scalars.num_float > 0 {
            inputs.push(InputInfo::Scalar {
                elem: gpu::Elem::Float,
                size: self.scalars.num_float,
            })
        }

        if self.scalars.num_uint > 0 {
            inputs.push(InputInfo::Scalar {
                elem: gpu::Elem::UInt,
                size: self.scalars.num_uint,
            })
        }

        if self.scalars.num_int > 0 {
            inputs.push(InputInfo::Scalar {
                elem: gpu::Elem::Int,
                size: self.scalars.num_int,
            })
        }

        CompilationInfo {
            inputs,
            outputs,
            scope: self.scope.clone(),
        }
    }
}
