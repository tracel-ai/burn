use burn_tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// A view of a tensor that can lazily produce TensorData
/// Stores a cloned tensor (cheap due to reference counting)
pub struct TensorView {
    /// Function to get tensor data when needed
    data_fn: Box<dyn Fn() -> TensorData>,
}

impl TensorView {
    /// Create a new tensor view from a float tensor
    pub fn from_float<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
        }
    }

    /// Create a new tensor view from an int tensor
    pub fn from_int<B: Backend, const D: usize>(tensor: &Tensor<B, D, Int>) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
        }
    }

    /// Create a new tensor view from a bool tensor
    pub fn from_bool<B: Backend, const D: usize>(tensor: &Tensor<B, D, Bool>) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
        }
    }

    /// Convert to TensorData (this is where actual data copy happens)
    pub fn to_data(&self) -> TensorData {
        (self.data_fn)()
    }
}
