use crate::module::ParamId;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use burn_tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// A lightweight view of a tensor that can lazily produce TensorData.
///
/// TensorView stores a cloned tensor internally (which is cheap due to reference counting)
/// and only materializes the actual data when `to_data()` is called. This allows
/// efficient inspection of module structure without the overhead of copying all tensor data.
pub struct TensorView {
    /// Function to get tensor data when needed
    data_fn: Box<dyn Fn() -> TensorData>,
    /// Path stack representing the module hierarchy
    pub path_stack: Vec<String>,
    /// Container stack representing the container types at each level
    pub container_stack: Vec<String>,
    /// Unique identifier for the tensor parameter
    pub tensor_id: ParamId,
}

impl TensorView {
    /// Create a new tensor view from a float tensor
    pub fn from_float<B: Backend, const D: usize>(
        tensor: &Tensor<B, D>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
            path_stack,
            container_stack,
            tensor_id,
        }
    }

    /// Create a new tensor view from an int tensor
    pub fn from_int<B: Backend, const D: usize>(
        tensor: &Tensor<B, D, Int>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
            path_stack,
            container_stack,
            tensor_id,
        }
    }

    /// Create a new tensor view from a bool tensor
    pub fn from_bool<B: Backend, const D: usize>(
        tensor: &Tensor<B, D, Bool>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Box::new(move || tensor.to_data()),
            path_stack,
            container_stack,
            tensor_id,
        }
    }

    /// Convert to TensorData (this is where actual data copy happens)
    pub fn to_data(&self) -> TensorData {
        (self.data_fn)()
    }

    /// Get the full path by joining the path stack
    pub fn full_path(&self) -> String {
        self.path_stack.join(".")
    }

    /// Get the full container path by joining the container stack
    pub fn container_path(&self) -> String {
        self.container_stack.join(".")
    }

    /// Get the immediate container type (last in the container stack)
    pub fn container_type(&self) -> String {
        self.container_stack
            .last()
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
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

        let view = TensorView::from_float(
            &tensor,
            vec!["test".to_string(), "weight".to_string()],
            vec!["TestModule".to_string(), "Param".to_string()],
            ParamId::new(),
        );
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 2], [3, 4]], &device);

        let view = TensorView::from_int(
            &tensor,
            vec!["test".to_string(), "int".to_string()],
            vec!["TestModule".to_string(), "Param".to_string()],
            ParamId::new(),
        );
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_bool() {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false], [false, true]], &device);

        let view = TensorView::from_bool(
            &tensor,
            vec!["test".to_string(), "bool".to_string()],
            vec!["TestModule".to_string(), "Param".to_string()],
            ParamId::new(),
        );
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }
}
