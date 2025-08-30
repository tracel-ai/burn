use burn_tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// A view of a tensor that can lazily produce TensorData
pub struct TensorView<'a> {
    /// Function to get tensor data when needed
    data_fn: Box<dyn Fn() -> TensorData + 'a>,
}

impl<'a> TensorView<'a> {
    /// Create a new tensor view from a float tensor
    pub fn from_float<B: Backend, const D: usize>(tensor: &'a Tensor<B, D>) -> Self {
        Self {
            data_fn: Box::new(move || tensor.to_data()),
        }
    }

    /// Create a new tensor view from an int tensor
    pub fn from_int<B: Backend, const D: usize>(tensor: &'a Tensor<B, D, Int>) -> Self {
        Self {
            data_fn: Box::new(move || tensor.to_data()),
        }
    }

    /// Create a new tensor view from a bool tensor
    pub fn from_bool<B: Backend, const D: usize>(tensor: &'a Tensor<B, D, Bool>) -> Self {
        Self {
            data_fn: Box::new(move || tensor.to_data()),
        }
    }

    /// Convert to TensorData (this is where actual data copy happens)
    pub fn to_data(&self) -> TensorData {
        (self.data_fn)()
    }
}
