use super::Scalars;
use crate::codegen::{dialect::gpu, InplaceMapping, Input, Output};
use burn_fusion::TensorDescription;

#[derive(Clone)]
pub struct CompilingInfo {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub scope: gpu::Scope,
    pub mappings: Vec<InplaceMapping>,
}

pub struct RunningInfo<'a> {
    pub inputs: Vec<&'a TensorDescription>,
    pub outputs: Vec<&'a TensorDescription>,
    pub scalars: &'a Scalars,
}
