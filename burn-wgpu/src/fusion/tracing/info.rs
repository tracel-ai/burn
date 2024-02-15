use super::Scalars;
use burn_fusion::TensorDescription;

pub struct ExecutionInfo<'a> {
    pub inputs: Vec<&'a TensorDescription>,
    pub outputs: Vec<&'a TensorDescription>,
    pub scalars: &'a Scalars,
}
