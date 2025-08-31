use alloc::boxed::Box;
use burn_tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// A lightweight view of a tensor that can lazily produce TensorData.
///
/// TensorView stores a cloned tensor internally (which is cheap due to reference counting)
/// and only materializes the actual data when `to_data()` is called. This allows
/// efficient inspection of module structure without the overhead of copying all tensor data.
///
/// # Examples
///
/// ```ignore
/// // Create a view from a tensor
/// let tensor = Tensor::<Backend, 2>::ones([3, 4], &device);
/// let view = TensorView::from_float(&tensor);
///
/// // Data is not copied until explicitly requested
/// let data = view.to_data();
/// assert_eq!(data.shape, vec![3, 4]);
///
/// // Views can be created from different tensor types
/// let int_tensor = Tensor::<Backend, 1, Int>::arange(0..10, &device);
/// let int_view = TensorView::from_int(&int_tensor);
///
/// let bool_tensor = Tensor::<Backend, 2, Bool>::zeros([2, 2], &device);
/// let bool_view = TensorView::from_bool(&bool_tensor);
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_tensor_view_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let view = TensorView::from_float(&tensor);
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 2], [3, 4]], &device);

        let view = TensorView::from_int(&tensor);
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_view_bool() {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false], [false, true]], &device);

        let view = TensorView::from_bool(&tensor);
        let data = view.to_data();

        assert_eq!(data.shape, vec![2, 2]);
    }
}
