use alloc::{format, string::String, vec, vec::Vec};
use burn_core::tensor::DType;

pub(crate) enum TensorCheck {
    Ok,
    Failed(FailedTensorCheck),
}

impl TensorCheck {
    fn register(self, ops: &str, error: TensorError) -> Self {
        let errors = match self {
            Self::Ok => vec![error],
            Self::Failed(mut failed) => {
                failed.errors.push(error);
                failed.errors
            }
        };
        Self::Failed(FailedTensorCheck {
            ops: ops.into(),
            errors,
        })
    }

    pub(crate) fn diag<const D: usize, const DO: usize>() -> Self {
        let mut check = Self::Ok;
        if D < 2 {
            check = check.register(
                "Diag",
                TensorError::new("Diagonal operations require tensors with at least 2 dimensions."),
            );
        }
        if DO != D.saturating_sub(1) {
            check = check.register(
                "Diag",
                TensorError::new("Output rank must be input rank minus 1 for diagonal"),
            );
        }
        check
    }

    pub(crate) fn lu_generic_param<const D: usize, const D1: usize>(ops: &str) -> Self {
        if D.checked_sub(1) != Some(D1) {
            return Self::Ok.register(
                ops,
                TensorError::new(
                    "D - 1 = D1 must hold for the generic parameters of LU decomposition.",
                ),
            );
        }
        Self::Ok
    }

    pub(crate) fn lu_input_tensor<const D: usize>(ops: &str, dims: &[usize], dtype: DType) -> Self {
        let mut check = Self::Ok;
        if matches!(dtype, DType::QFloat(_)) {
            check = check.register(
                ops,
                TensorError::new("The input tensor must have a real float dtype"),
            );
        }
        if dims.len() < 2 || D < 2 {
            check = check.register(
                ops,
                TensorError::new("The input tensor must have at least two dimensions."),
            );
        }
        check
    }

    pub(crate) fn qr_input_tensor<const D: usize>(ops: &str, dims: &[usize], dtype: DType) -> Self {
        Self::lu_input_tensor::<D>(ops, dims, dtype)
    }

    pub(crate) fn svd_input_tensor<const D: usize, const D1: usize>(
        ops: &str,
        dims: &[usize],
        dtype: DType,
    ) -> Self {
        let mut check = Self::Ok;
        if matches!(dtype, DType::QFloat(_)) {
            check = check.register(
                ops,
                TensorError::new("The input tensor must have a real float dtype"),
            );
        }
        if dims.len() < 2 || D < 2 {
            check = check.register(
                ops,
                TensorError::new(
                    "The input tensor for SVD decomposition must have at least two dimensions.",
                ),
            );
        }
        if D.checked_sub(1) != Some(D1) {
            check = check.register(
                ops,
                TensorError::new("D - 1 = D1 must hold for the generic parameters of linalg::svd."),
            );
        }
        check
    }

    pub(crate) fn det<const D: usize, const D1: usize, const D2: usize>(
        dims: [usize; D],
        dtype: DType,
    ) -> Self {
        let mut check = Self::lu_input_tensor::<D>("det", &dims, dtype);
        if D.checked_sub(1) != Some(D1) {
            check = check.register(
                "det",
                TensorError::new(
                    "D - 1 = D1 must hold for the generic parameters of the linalg::det function.",
                ),
            );
        }
        if D.checked_sub(2) != Some(D2) {
            check = check.register(
                "det",
                TensorError::new("The output tensor rank must be less than input tensor rank by 2"),
            );
        }
        if D < 3 {
            check = check.register(
                "det",
                TensorError::new("The input tensor must have at least 3 dimensions."),
            );
        } else if dims[D - 1] != dims[D - 2] {
            check = check.register(
                "det",
                TensorError::new("The last two dimensions of the input tensor must be equal."),
            );
        }
        check
    }
}

pub(crate) struct FailedTensorCheck {
    ops: String,
    errors: Vec<TensorError>,
}

impl FailedTensorCheck {
    pub(crate) fn format(self) -> String {
        self.errors.into_iter().enumerate().fold(
            format!(
                "=== Tensor Operation Error ===\n  Operation: '{}'\n  Reason:",
                self.ops
            ),
            |mut message, (index, error)| {
                message += &format!("\n    {}. {}", index + 1, error.description);
                message
            },
        ) + "\n"
    }
}

struct TensorError {
    description: String,
}

impl TensorError {
    fn new(description: &str) -> Self {
        Self {
            description: description.into(),
        }
    }
}

#[track_caller]
pub(crate) fn unwrap_dim_index<E>(result: Result<usize, E>, op: &str) -> usize
where
    E: core::fmt::Display,
{
    match result {
        Ok(dim) => dim,
        Err(error) => panic!("{op}: {error}"),
    }
}
