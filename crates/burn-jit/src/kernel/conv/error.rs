use cubecl::{linalg::matmul::kernels::MatmulLaunchError, tune::AutotuneError};

#[derive(Debug)]
pub enum ConvLaunchError {
    Matmul(MatmulLaunchError),
    Unknown,
}

impl From<MatmulLaunchError> for ConvLaunchError {
    fn from(value: MatmulLaunchError) -> Self {
        Self::Matmul(value)
    }
}

#[allow(clippy::from_over_into)]
impl Into<AutotuneError> for ConvLaunchError {
    fn into(self) -> AutotuneError {
        AutotuneError::Unknown(format!("{self:?}"))
    }
}
