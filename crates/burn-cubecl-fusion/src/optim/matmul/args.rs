use crate::engine::codegen::ir::{FuseArg, FuseType};
use cubecl::quant::scheme::QuantScheme;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
/// Argument to a matmul operation.
pub enum MatmulArg {
    Normal(FuseArg),
    Quantized {
        data: FuseArg,
        scales: FuseArg,
        precision: FuseType,
        scheme: QuantScheme,
    },
}

impl MatmulArg {
    pub fn data(&self) -> &FuseArg {
        match self {
            MatmulArg::Normal(arg) => arg,
            MatmulArg::Quantized { data, .. } => data,
        }
    }

    pub fn scheme(&self) -> Option<&QuantScheme> {
        match self {
            MatmulArg::Normal(_) => None,
            MatmulArg::Quantized { scheme, .. } => Some(scheme),
        }
    }

    pub fn precision(&self) -> FuseType {
        match self {
            MatmulArg::Normal(arg) => arg.precision(),
            MatmulArg::Quantized { precision, .. } => *precision,
        }
    }
}
