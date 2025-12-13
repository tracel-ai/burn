use crate::{LibTorchDevice, TchElement};
use burn_backend::{DType, FloatDType, IntDType, Shape, TensorData, TensorMetadata};
use libc::c_void;
use std::sync::Arc;

/// A reference to a tensor storage.
///
/// We manually implement `Sync` and `Send` unsafely, so even if we could use `Rc`, it isn't safe.
#[allow(clippy::arc_with_non_send_sync)]
pub type StorageRef = Arc<*mut c_void>;

/// A reference to a tensor storage.
#[derive(PartialEq, Debug, Clone)]
pub enum Storage {
    /// When a tensor is a partial view of another tensor.
    View {
        /// Storage reference for the whole buffer.
        buffer_ref: StorageRef,
        /// Storage reference for the partial buffer.
        view_ref: StorageRef,
    },
    /// When a tensor use all of its buffer.
    Owned {
        /// Storage reference for the whole buffer.
        buffer_ref: StorageRef,
    },
}

impl Storage {
    /// Check if the storage can be used inplace.
    pub fn can_mut(&self) -> bool {
        match self {
            Storage::View {
                buffer_ref: start_ref,
                view_ref,
            } => Arc::strong_count(start_ref) == 1 && Arc::strong_count(view_ref) == 1,
            Storage::Owned {
                buffer_ref: start_ref,
            } => Arc::strong_count(start_ref) == 1,
        }
    }

    /// Get the whole buffer reference.
    pub fn buffer_ref(&self) -> &StorageRef {
        match self {
            Storage::View {
                buffer_ref: start_ref,
                view_ref: _,
            } => start_ref,
            Storage::Owned {
                buffer_ref: start_ref,
            } => start_ref,
        }
    }
}

/// A tensor using the tch backend.
#[derive(Debug, PartialEq)]
pub struct TchTensor {
    /// Handle to the tensor. Call methods on this field.
    pub tensor: tch::Tensor,

    /// The tensor's storage
    pub storage: Storage,
}

impl TensorMetadata for TchTensor {
    fn dtype(&self) -> DType {
        match self.tensor.kind() {
            tch::Kind::Uint8 => DType::U8,
            tch::Kind::Int8 => DType::I8,
            tch::Kind::Int16 => DType::I16,
            tch::Kind::Int => DType::I32,
            tch::Kind::Int64 => DType::I64,
            tch::Kind::Half => DType::F16,
            tch::Kind::Float => DType::F32,
            tch::Kind::Double => DType::F64,
            tch::Kind::Bool => DType::Bool,
            tch::Kind::BFloat16 => DType::BF16,
            // Complex and quantization types are not valid/implemented.
            _ => unimplemented!(),
        }
    }

    fn shape(&self) -> Shape {
        Shape::from(self.tensor.size())
    }

    fn rank(&self) -> usize {
        self.tensor.dim()
    }
}

impl burn_backend::QTensorPrimitive for TchTensor {
    fn scheme(&self) -> &burn_backend::quantization::QuantScheme {
        unimplemented!("Quantization is not supported")
    }
}

impl core::fmt::Display for TchTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

pub(crate) trait IntoKind {
    fn try_into_kind(self) -> Result<tch::Kind, tch::TchError>;
    fn into_kind(self) -> tch::Kind
    where
        Self: Sized,
    {
        self.try_into_kind().unwrap()
    }
}

impl IntoKind for IntDType {
    fn try_into_kind(self) -> Result<tch::Kind, tch::TchError> {
        let dtype: DType = self.into();
        dtype.try_into_kind()
    }
}

impl IntoKind for FloatDType {
    fn try_into_kind(self) -> Result<tch::Kind, tch::TchError> {
        let dtype: DType = self.into();
        dtype.try_into_kind()
    }
}

impl IntoKind for DType {
    fn try_into_kind(self) -> Result<tch::Kind, tch::TchError> {
        match self {
            DType::F64 => Ok(tch::Kind::Double),
            DType::F32 => Ok(tch::Kind::Float),
            DType::Flex32 => Ok(tch::Kind::Float),
            DType::F16 => Ok(tch::Kind::Half),
            DType::BF16 => Ok(tch::Kind::BFloat16),
            DType::I64 => Ok(tch::Kind::Int64),
            DType::I32 => Ok(tch::Kind::Int),
            DType::I16 => Ok(tch::Kind::Int16),
            DType::I8 => Ok(tch::Kind::Int8),
            DType::U8 => Ok(tch::Kind::Uint8),
            DType::Bool => Ok(tch::Kind::Bool),
            other => Err(tch::TchError::Kind(format!("Unsupported dtype {other:?}"))),
        }
    }
}

impl TchTensor {
    /// Create a new tensor.
    ///
    /// Note that if the tensor was created from an operation that may reuse the same tensor
    /// storage as the parent, you should use [from_existing](TchTensor::from_existing)
    /// instead.
    pub fn new(tensor: tch::Tensor) -> Self {
        #[allow(clippy::arc_with_non_send_sync)]
        let storage = Storage::Owned {
            buffer_ref: Arc::new(tensor.data_ptr()),
        };

        Self { tensor, storage }
    }

    /// Create a tensor that was created from an operation executed on a parent tensor.
    ///
    /// If the child tensor shared the same storage as its parent, it will be cloned, effectively
    /// tracking how much tensors point to the same memory space.
    pub fn from_existing(tensor: tch::Tensor, storage_parent: Storage) -> Self {
        let storage_child = tensor.data_ptr();
        let mut is_a_new_tensor = true;

        match &storage_parent {
            Storage::View {
                buffer_ref: start_ref,
                view_ref,
            } => {
                if storage_child == *start_ref.as_ref() || storage_child == *view_ref.as_ref() {
                    is_a_new_tensor = false;
                }
            }
            Storage::Owned {
                buffer_ref: start_ref,
            } => {
                if storage_child == *start_ref.as_ref() {
                    is_a_new_tensor = false;
                }
            }
        };

        let storage = match is_a_new_tensor {
            true => Storage::Owned {
                #[allow(clippy::arc_with_non_send_sync)]
                buffer_ref: Arc::new(storage_child),
            },
            false => storage_parent.clone(),
        };

        Self { tensor, storage }
    }

    /// Create a tensor that uses a part of its parent tensor such as slice and narrow.
    pub fn partial(tensor: tch::Tensor, storage_parent: Storage) -> Self {
        let storage = Storage::View {
            buffer_ref: storage_parent.buffer_ref().clone(),
            #[allow(clippy::arc_with_non_send_sync)]
            view_ref: Arc::new(tensor.data_ptr()),
        };
        Self { tensor, storage }
    }
}

// This is safe since we don't use autodiff from LibTorch.
// Also, atomic reference counting is used to know if the tensor's data can be reused.
// If there are multiple reference on the same tensor, it becomes read only.
unsafe impl Send for TchTensor {}
unsafe impl Sync for TchTensor {}

impl TchTensor {
    /// Checks if the tensor can be mutated in-place.
    ///
    /// Returns `true` if the tensor's stride does not contain zero (no broadcasting)
    /// and the storage can be mutated.
    pub fn can_mut(&self) -> bool {
        let stride_contains_zero = self.tensor.stride().contains(&0);

        !stride_contains_zero && self.storage.can_mut()
    }

    /// Executes an operation on a tensor if the data can be reused.
    pub fn mut_ops<F: Fn(&mut tch::Tensor) -> tch::Tensor>(
        &mut self,
        func: F,
    ) -> Option<TchTensor> {
        if !self.can_mut() {
            return None;
        }

        let data = self.storage.clone();
        Some(TchTensor::from_existing(func(&mut self.tensor), data))
    }

    /// Executes a unary operation, reusing the tensor data if possible.
    pub fn unary_ops<FOwn, FRef>(self, fown: FOwn, fref: FRef) -> TchTensor
    where
        FOwn: Fn(tch::Tensor) -> tch::Tensor,
        FRef: Fn(&tch::Tensor) -> tch::Tensor,
    {
        if !self.can_mut() {
            return TchTensor::from_existing(fref(&self.tensor), self.storage);
        }

        TchTensor::from_existing(fown(self.tensor), self.storage)
    }

    /// Executes a binary operation, reusing the tensor data if possible.
    pub fn binary_ops_tensor<FLMut, FRMut, FRef>(
        mut lhs: Self,
        mut rhs: Self,
        flmut: FLMut,
        frmut: FRMut,
        fref: FRef,
    ) -> TchTensor
    where
        FLMut: Fn(&mut tch::Tensor, &tch::Tensor) -> tch::Tensor,
        FRMut: Fn(&tch::Tensor, &mut tch::Tensor) -> tch::Tensor,
        FRef: Fn(&tch::Tensor, &tch::Tensor) -> tch::Tensor,
    {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        // Both lhs and rhs are expected to have the same rank
        let d_out = lhs_shape.num_dims();
        let mut out_shape = Shape::from(vec![1usize; d_out]);

        for i in 0..d_out {
            out_shape[i] = usize::max(lhs_shape[i], rhs_shape[i]);
        }

        let num_elements_out = out_shape.num_elements();

        // Attempt to mutate lhs tensor
        if lhs_shape.num_elements() == num_elements_out
            && let Some(output) = lhs.mut_ops(|lhs| flmut(lhs, &rhs.tensor))
        {
            return output;
        }

        // Attempt to mutate rhs tensor
        if rhs_shape.num_elements() == num_elements_out
            && let Some(output) = rhs.mut_ops(|rhs| frmut(&lhs.tensor, rhs))
        {
            return output;
        }

        let storage = lhs.storage;
        let tensor = fref(&lhs.tensor, &rhs.tensor);

        TchTensor::from_existing(tensor, storage)
    }
}

impl Clone for TchTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.shallow_clone(),
            storage: self.storage.clone(),
        }
    }
}

/// A shape that can be used by LibTorch.
#[derive(Debug)]
pub struct TchShape {
    /// The shape's dimensions.
    pub dims: Vec<i64>,
}

impl From<Shape> for TchShape {
    fn from(shape: Shape) -> Self {
        TchShape {
            dims: shape.into_iter().map(|d| d as i64).collect(),
        }
    }
}

impl From<&[usize]> for TchShape {
    fn from(shape: &[usize]) -> Self {
        TchShape {
            dims: shape.iter().map(|d| *d as i64).collect(),
        }
    }
}

impl TchTensor {
    /// Creates a new tensor from a shape and a device.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// A new tensor.
    pub fn from_data<E: TchElement>(data: TensorData, device: tch::Device) -> Self {
        let shape_tch = TchShape::from(data.shape.as_slice());
        let tensor = tch::Tensor::from_slice(data.as_slice::<E>().unwrap()).to(device);
        let tensor = tensor.reshape(shape_tch.dims).to_kind(E::kind());

        Self::new(tensor)
    }
}

impl TchTensor {
    /// Creates an empty tensor from a shape and a device.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A new empty tensor.
    pub fn empty<E: TchElement>(shape: Shape, device: LibTorchDevice) -> Self {
        let shape_tch = TchShape::from(shape);
        let tensor = tch::Tensor::empty(shape_tch.dims, (E::kind(), device.into()));

        Self::new(tensor)
    }
}

// Adapted from `tch` to use patched `T::kind()` instead of `T::KIND` which is incorrect for bf16.
// TODO: remove when fixed in `tch` release (https://github.com/LaurentMazare/tch-rs/pull/996).
impl<T: TchElement + Copy> TryFrom<&TchTensor> for Vec<T> {
    type Error = tch::TchError;
    fn try_from(tensor: &TchTensor) -> Result<Self, Self::Error> {
        let tensor = &tensor.tensor;
        let size = tensor.size();
        if size.len() != 1 {
            Err(tch::TchError::Convert(format!(
                "Attempting to convert a Tensor with {} dimensions to flat vector",
                size.len()
            )))?;
        }
        let numel = size[0] as usize;
        let mut vec = vec![T::ZERO; numel];
        // Adapted to use patched `T::kind()` instead
        // TODO: tensor.f_to_kind(T::KIND)?.f_copy_data(&mut vec, numel)?;
        f_copy_data(&mut tensor.f_to_kind(T::kind())?, &mut vec, numel)?;
        Ok(vec)
    }
}

unsafe fn ptr_to_string(ptr: *mut libc::c_char) -> Option<String> {
    if !ptr.is_null() {
        unsafe {
            let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            libc::free(ptr as *mut libc::c_void);
            Some(str)
        }
    } else {
        None
    }
}

/// Copies `numel` elements from `self` to `dst`.
fn f_copy_data<T: TchElement>(
    tensor: &mut tch::Tensor,
    dst: &mut [T],
    numel: usize,
) -> Result<(), tch::TchError> {
    if T::kind() != tensor.f_kind()? {
        return Err(tch::TchError::Kind(format!(
            "incoherent elt kind, {:?} != {:?}",
            tensor.f_kind(),
            T::kind()
        )));
    }
    if dst.len() < numel {
        return Err(tch::TchError::Shape(format!("slice len < {numel}")));
    }

    unsafe {
        torch_sys::at_copy_data(
            tensor.as_mut_ptr(),
            dst.as_mut_ptr() as *const c_void,
            numel,
            T::kind().elt_size_in_bytes(),
        );
        match ptr_to_string(torch_sys::get_and_reset_last_err()) {
            None => Ok(()),
            Some(c_error) => Err(tch::TchError::Torch(c_error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::ops::FloatTensorOps;
    use burn_backend::{Backend, quantization::QuantScheme, read_sync};

    type B = crate::LibTorch<f32>;

    #[test]
    fn should_have_bf16_kind() {
        let data = TensorData::from([4.0, 4.0]);
        let tensor_1: TchTensor = B::float_from_data(data, &Default::default());
        let tensor_2 = B::float_cast(tensor_1, DType::BF16.into());

        assert_eq!(tensor_2.tensor.kind(), tch::Kind::BFloat16);

        let out = read_sync(B::float_into_data(tensor_2)).unwrap();

        out.assert_eq(&TensorData::from([4.0, 4.0]), false);
    }

    #[test]
    fn should_support_dtypes() {
        let device = Default::default();

        assert!(B::supports_dtype(&device, DType::F64));
        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::Flex32));
        assert!(B::supports_dtype(&device, DType::F16));
        assert!(B::supports_dtype(&device, DType::BF16));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(&device, DType::Bool));

        assert!(!B::supports_dtype(&device, DType::U64));
        assert!(!B::supports_dtype(&device, DType::U32));
        assert!(!B::supports_dtype(&device, DType::U16));
        assert!(!B::supports_dtype(
            &device,
            DType::QFloat(QuantScheme::default())
        ));
    }
}

unsafe extern "C" {
    /// Dummy function to get CUDA to link properly
    pub fn dummy_cuda_dependency();
}

#[used]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [dummy_cuda_dependency];
