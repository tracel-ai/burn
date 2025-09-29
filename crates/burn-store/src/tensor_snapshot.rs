use alloc::rc::Rc;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_tensor::{Bool, Int, Tensor, TensorData, backend::Backend};

/// Error type for TensorSnapshot operations
#[derive(Debug, Clone)]
pub enum TensorSnapshotError {
    /// I/O error occurred while loading tensor data
    IoError(String),
    /// Data corruption or invalid format
    DataError(String),
    /// Panic occurred while loading tensor data
    PanicError(String),
}

/// A lightweight snapshot of a tensor that can lazily produce TensorData.
///
/// TensorSnapshot stores a cloned tensor internally (which is cheap due to reference counting)
/// and only materializes the actual data when `to_data()` is called. This allows
/// efficient inspection of module structure without the overhead of copying all tensor data.
///
/// The dtype and shape are cached for efficient access without requiring data materialization,
/// which is particularly useful for serialization formats that need metadata upfront.
pub struct TensorSnapshot {
    /// Function to get tensor data when needed (Rc allows cloning)
    data_fn: Rc<dyn Fn() -> Result<TensorData, TensorSnapshotError>>,
    /// Data type of the tensor (cached for efficient access)
    pub dtype: burn_tensor::DType,
    /// Shape of the tensor (cached for efficient access)
    pub shape: Vec<usize>,
    /// Path stack representing the module hierarchy
    pub path_stack: Option<Vec<String>>,
    /// Container stack representing the container types at each level
    pub container_stack: Option<Vec<String>>,
    /// Unique identifier for the tensor parameter
    pub tensor_id: Option<ParamId>,
}

impl TensorSnapshot {
    /// Create a new tensor snapshot from a float tensor
    pub fn from_float<B: Backend, const D: usize>(
        tensor: &Tensor<B, D>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let dtype = tensor.dtype();
        let shape = tensor.shape().dims.to_vec();
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Rc::new(move || Ok(tensor.to_data())),
            dtype,
            shape,
            path_stack: Some(path_stack),
            container_stack: Some(container_stack),
            tensor_id: Some(tensor_id),
        }
    }

    /// Create a new tensor snapshot from an int tensor
    pub fn from_int<B: Backend, const D: usize>(
        tensor: &Tensor<B, D, Int>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let dtype = tensor.dtype();
        let shape = tensor.shape().dims.to_vec();
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Rc::new(move || Ok(tensor.to_data())),
            dtype,
            shape,
            path_stack: Some(path_stack),
            container_stack: Some(container_stack),
            tensor_id: Some(tensor_id),
        }
    }

    /// Create a new tensor snapshot from a bool tensor
    pub fn from_bool<B: Backend, const D: usize>(
        tensor: &Tensor<B, D, Bool>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let dtype = tensor.dtype();
        let shape = tensor.shape().dims.to_vec();
        let tensor = tensor.clone(); // Clone is cheap (reference counted)
        Self {
            data_fn: Rc::new(move || Ok(tensor.to_data())),
            dtype,
            shape,
            path_stack: Some(path_stack),
            container_stack: Some(container_stack),
            tensor_id: Some(tensor_id),
        }
    }

    /// Convert to TensorData (this is where actual data copy happens)
    #[cfg(feature = "std")]
    pub fn to_data(&self) -> Result<TensorData, TensorSnapshotError> {
        // Use AssertUnwindSafe since we're working with Rc which is not UnwindSafe
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (self.data_fn)())).unwrap_or_else(
            |_| {
                Err(TensorSnapshotError::PanicError(
                    "Panic occurred while loading tensor data".to_string(),
                ))
            },
        )
    }

    /// Convert to TensorData (this is where actual data copy happens)
    #[cfg(not(feature = "std"))]
    pub fn to_data(&self) -> Result<TensorData, TensorSnapshotError> {
        (self.data_fn)() // Can't catch panics in no-std, do it when core::panic::AssertUnwindSafe is available
    }

    /// Get the full path by joining the path stack
    pub fn full_path(&self) -> String {
        self.path_stack
            .as_ref()
            .map(|stack| stack.join("."))
            .unwrap_or_default()
    }

    /// Get the full container path by joining the container stack
    pub fn container_path(&self) -> String {
        self.container_stack
            .as_ref()
            .map(|stack| stack.join("."))
            .unwrap_or_default()
    }

    /// Get the immediate container type (last in the container stack)
    pub fn container_type(&self) -> String {
        self.container_stack
            .as_ref()
            .and_then(|stack| stack.last())
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    }

    /// Create a TensorSnapshot from a closure that produces TensorData
    /// This is used internally for lazy loading
    pub fn from_closure(
        data_fn: Rc<dyn Fn() -> Result<TensorData, TensorSnapshotError>>,
        dtype: burn_tensor::DType,
        shape: Vec<usize>,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        Self {
            data_fn,
            dtype,
            shape,
            path_stack: Some(path_stack),
            container_stack: Some(container_stack),
            tensor_id: Some(tensor_id),
        }
    }

    /// Create a TensorSnapshot from TensorData directly
    pub fn from_data(
        data: TensorData,
        path_stack: Vec<String>,
        container_stack: Vec<String>,
        tensor_id: ParamId,
    ) -> Self {
        let dtype = data.dtype;
        let shape = data.shape.clone();
        Self {
            data_fn: Rc::new(move || Ok(data.clone())),
            dtype,
            shape,
            path_stack: Some(path_stack),
            container_stack: Some(container_stack),
            tensor_id: Some(tensor_id),
        }
    }

    /// Get the size of the tensor data in bytes without materializing it
    pub fn data_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size()
    }

    /// Clone the data function for lazy composition
    pub fn clone_data_fn(&self) -> Rc<dyn Fn() -> Result<TensorData, TensorSnapshotError>> {
        self.data_fn.clone()
    }
}

impl Clone for TensorSnapshot {
    fn clone(&self) -> Self {
        // Clone lazily - keep the same data function
        Self {
            data_fn: self.data_fn.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
            path_stack: self.path_stack.clone(),
            container_stack: self.container_stack.clone(),
            tensor_id: self.tensor_id,
        }
    }
}

impl core::fmt::Debug for TensorSnapshot {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorSnapshot")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("path_stack", &self.path_stack)
            .field("container_stack", &self.container_stack)
            .field("tensor_id", &self.tensor_id)
            .finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    type TestBackend = burn_ndarray::NdArray;
    use alloc::string::ToString;
    use burn_tensor::DType;

    #[test]
    fn tensor_view_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);

        let snapshot = TensorSnapshot::from_float(
            &tensor,
            vec!["test".to_string(), "weight".to_string()],
            vec!["TestModule".to_string(), "Param".to_string()],
            ParamId::new(),
        );

        // Test metadata access without materialization
        assert_eq!(snapshot.dtype, DType::F32);
        assert_eq!(snapshot.shape, vec![2, 2]);
        assert_eq!(snapshot.full_path(), "test.weight");
        assert_eq!(snapshot.container_path(), "TestModule.Param");

        // Test data materialization
        let data = snapshot.to_data().unwrap();
        assert_eq!(data.shape, vec![2, 2]);
        assert_eq!(data.dtype, DType::F32);
    }

    #[test]
    fn tensor_view_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2, Int>::from_data([[1, 2], [3, 4]], &device);

        let snapshot = TensorSnapshot::from_int(
            &tensor,
            vec!["test".to_string(), "int".to_string()],
            vec!["TestModule".to_string(), "Param".to_string()],
            ParamId::new(),
        );

        // Test metadata access without materialization
        // TestBackend uses I64 for integers
        assert_eq!(snapshot.dtype, DType::I64);
        assert_eq!(snapshot.shape, vec![2, 2]);

        let data = snapshot.to_data().unwrap();
        assert_eq!(data.shape, vec![2, 2]);
        assert_eq!(data.dtype, DType::I64);
    }

    #[test]
    fn tensor_view_bool() {
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false], [false, true]], &device);

        let snapshot = TensorSnapshot::from_bool(
            &tensor,
            vec!["test".to_string(), "bool".to_string()],
            vec!["TestModule".to_string(), "Param".to_string()],
            ParamId::new(),
        );

        // Test metadata access without materialization
        assert_eq!(snapshot.dtype, DType::Bool);
        assert_eq!(snapshot.shape, vec![2, 2]);

        let data = snapshot.to_data().unwrap();
        assert_eq!(data.shape, vec![2, 2]);
        assert_eq!(data.dtype, DType::Bool);
    }

    #[test]
    fn data_len() {
        let device = Default::default();

        // Test F32 tensor (4 bytes per element)
        let tensor_f32 = Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let view_f32 = TensorSnapshot::from_float(
            &tensor_f32,
            vec!["test".to_string()],
            vec!["Module".to_string()],
            ParamId::new(),
        );
        assert_eq!(view_f32.data_len(), 16); // 4 elements * 4 bytes

        // Test I64 tensor (8 bytes per element) - TestBackend uses I64 for Int
        let tensor_i64 =
            Tensor::<TestBackend, 3, Int>::from_data([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
        let view_i64 = TensorSnapshot::from_int(
            &tensor_i64,
            vec!["test".to_string()],
            vec!["Module".to_string()],
            ParamId::new(),
        );
        assert_eq!(view_i64.data_len(), 64); // 8 elements * 8 bytes (I64)

        // Test Bool tensor (1 byte per element)
        let tensor_bool =
            Tensor::<TestBackend, 2, Bool>::from_data([[true, false], [false, true]], &device);
        let view_bool = TensorSnapshot::from_bool(
            &tensor_bool,
            vec!["test".to_string()],
            vec!["Module".to_string()],
            ParamId::new(),
        );
        assert_eq!(view_bool.data_len(), 4); // 4 elements * 1 byte
    }

    #[test]
    fn from_closure() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let dtype = data.dtype;
        let shape = data.shape.clone();

        let snapshot = TensorSnapshot::from_closure(
            Rc::new(move || Ok(data.clone())),
            dtype,
            shape.clone(),
            vec!["model".to_string(), "layer".to_string()],
            vec!["Model".to_string(), "Layer".to_string()],
            ParamId::new(),
        );

        // Test metadata access
        assert_eq!(snapshot.dtype, DType::F32);
        assert_eq!(snapshot.shape, vec![4]);
        assert_eq!(snapshot.full_path(), "model.layer");
        assert_eq!(snapshot.data_len(), 16); // 4 * 4 bytes

        // Test data materialization
        let materialized = snapshot.to_data().unwrap();
        assert_eq!(materialized.shape, vec![4]);
    }

    #[test]
    fn from_data() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let original_dtype = data.dtype;
        let original_shape = data.shape.clone();

        let snapshot = TensorSnapshot::from_data(
            data,
            vec!["encoder".to_string(), "weight".to_string()],
            vec!["Encoder".to_string(), "Dense".to_string()],
            ParamId::new(),
        );

        // Test metadata
        assert_eq!(snapshot.dtype, original_dtype);
        assert_eq!(snapshot.shape, original_shape);
        assert_eq!(snapshot.full_path(), "encoder.weight");
        assert_eq!(snapshot.container_type(), "Dense");
        assert_eq!(snapshot.data_len(), 24); // 6 * 4 bytes

        // Test data materialization
        let materialized = snapshot.to_data().unwrap();
        assert_eq!(materialized.shape, original_shape);
    }

    #[test]
    #[cfg(feature = "std")]
    fn panic_catching_in_to_data() {
        use alloc::rc::Rc;

        // Create a TensorSnapshot with a closure that panics
        let snapshot = TensorSnapshot {
            data_fn: Rc::new(|| panic!("Test panic in data_fn")),
            dtype: DType::F32,
            shape: vec![2, 2],
            path_stack: Some(vec!["test".to_string()]),
            container_stack: Some(vec!["Test".to_string()]),
            tensor_id: Some(ParamId::new()),
        };

        // When std is available, to_data should catch the panic and return an error
        let result = snapshot.to_data();
        assert!(result.is_err());

        match result {
            Err(TensorSnapshotError::PanicError(msg)) => {
                assert!(msg.contains("Panic occurred"));
            }
            _ => panic!("Expected PanicError with panic message"),
        }
    }

    #[test]
    fn error_propagation_in_closure() {
        use alloc::rc::Rc;

        // Create a snapshot with a closure that returns an error
        let snapshot = TensorSnapshot::from_closure(
            Rc::new(|| Err(TensorSnapshotError::IoError("Simulated IO error".into()))),
            DType::F32,
            vec![2, 2],
            vec!["error_test".into()],
            vec![],
            ParamId::new(),
        );

        // Should return an error when trying to get data
        let result = snapshot.to_data();
        assert!(result.is_err());
        match result {
            Err(TensorSnapshotError::IoError(msg)) => {
                assert!(msg.contains("Simulated IO error"));
            }
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn container_type_extraction() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1>::from_data([1.0, 2.0, 3.0], &device);

        let snapshot = TensorSnapshot::from_float(
            &tensor,
            vec![
                "model".to_string(),
                "layer1".to_string(),
                "weight".to_string(),
            ],
            vec![
                "Model".to_string(),
                "Conv2d".to_string(),
                "Param".to_string(),
            ],
            ParamId::new(),
        );

        assert_eq!(snapshot.container_type(), "Param");
        assert_eq!(snapshot.container_path(), "Model.Conv2d.Param");
        assert_eq!(snapshot.full_path(), "model.layer1.weight");
    }
}
