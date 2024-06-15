use super::Scalars;
use burn_cube::{
    ir::{Elem, FloatKind, IntKind, Item, Scope, Variable, Visibility},
    InputInfo, KernelExpansion, OutputInfo,
};
use burn_tensor::repr::TensorDescription;
use serde::{Deserialize, Serialize};

/// A trace encapsulates all information necessary to perform the compilation and execution of
/// captured [tensor operations](burn_tensor::repr::OperationDescription).
///
/// A trace should be built using a [builder](super::TraceBuilder).
#[derive(new, Clone, Serialize, Deserialize)]
pub struct Trace {
    inputs: Vec<(TensorDescription, Elem, Variable)>,
    output_writes: Vec<(TensorDescription, Elem, Variable)>,
    locals: Vec<u16>,
    scalars: Scalars,
    scope: Scope,
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
            outputs: self.output_writes.iter().map(|a| &a.0).collect::<Vec<_>>(),
            scalars: &self.scalars,
        }
    }

    /// Collect information related to compiling the trace.
    pub fn compiling(&self) -> KernelExpansion {
        let mut inputs = self
            .inputs
            .iter()
            .map(|(_tensor, elem, _)| InputInfo::Array {
                item: Item::new(*elem),
                visibility: Visibility::Read,
            })
            .collect::<Vec<_>>();

        let outputs = self
            .output_writes
            .iter()
            .zip(self.locals.iter())
            .map(
                |((_tensor, elem, index_ref), local)| OutputInfo::ArrayWrite {
                    item: Item::new(*elem),
                    local: *local,
                    position: *index_ref,
                },
            )
            .collect::<Vec<_>>();

        // NOTE: we might want to pass a struct including all inputs/outputs metadata instead of 3 arrays
        if self.scalars.num_float > 0 {
            inputs.push(InputInfo::Scalar {
                elem: Elem::Float(FloatKind::F32),
                size: self.scalars.num_float,
            })
        }

        if self.scalars.num_uint > 0 {
            inputs.push(InputInfo::Scalar {
                elem: Elem::UInt,
                size: self.scalars.num_uint,
            })
        }

        if self.scalars.num_int > 0 {
            inputs.push(InputInfo::Scalar {
                elem: Elem::Int(IntKind::I32),
                size: self.scalars.num_int,
            })
        }

        KernelExpansion {
            inputs,
            outputs,
            scope: self.scope.clone(),
        }
    }
}
