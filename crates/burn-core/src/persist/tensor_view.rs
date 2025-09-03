use crate::module::ParamId;
use alloc::boxed::Box;
use alloc::string::String;
use burn_tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// A lightweight view of a tensor that can lazily produce TensorData.
///
/// TensorView stores a cloned tensor internally (which is cheap due to reference counting)
/// and only materializes the actual data when `to_data()` is called. This allows
/// efficient inspection of module structure without the overhead of copying all tensor data.
pub struct TensorView {
    /// Function to get tensor data when needed
    data_fn: Box<dyn Fn() -> TensorData>,
    /// Full path to the tensor in the module hierarchy
    pub full_path: String,
    /// Unique identifier for the tensor parameter
    pub tensor_id: ParamId,
}

impl TensorView {
    /// Create a new tensor view from a float tensor
    pub fn from_float<B: Backend, const D: usize>(
        tensor: &Tensor<B, D>,
        full_path: String,
        tensor_id: ParamId,
    ) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
            full_path,
            tensor_id,
        }
    }

    /// Create a new tensor view from an int tensor
    pub fn from_int<B: Backend, const D: usize>(
        tensor: &Tensor<B, D, Int>,
        full_path: String,
        tensor_id: ParamId,
    ) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
            full_path,
            tensor_id,
        }
    }

    /// Create a new tensor view from a bool tensor
    pub fn from_bool<B: Backend, const D: usize>(
        tensor: &Tensor<B, D, Bool>,
        full_path: String,
        tensor_id: ParamId,
    ) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
            full_path,
            tensor_id,
        }
    }

    /// Convert to TensorData (this is where actual data copy happens)
    pub fn to_data(&self) -> TensorData {
        (self.data_fn)()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::TestBackend;
    use alloc::string::ToString;

    #[test]
    fn test_tensor_view_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let view = TensorView::from_float(&tensor, "test.weight".to_string(), ParamId::new());
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 2], [3, 4]], &device);

        let view = TensorView::from_int(&tensor, "test.int".to_string(), ParamId::new());
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_bool() {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false], [false, true]], &device);

        let view = TensorView::from_bool(&tensor, "test.bool".to_string(), ParamId::new());
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }
}
